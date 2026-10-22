# Adapted from GAIC-Pytorch, https://github.com/bo-zhang-cs/GAIC-Pytorch
# Copyright (c) 2022 bo-zhang. Licensed under the MIT License.
# Copyright notice and full license are preserved in THIRD_PARTY_NOTICES.md.
# Modification: replace the upstream custom CUDA RoI/RoD operators with a
# batch-one FP32 sampling graph for CPU ONNX inference.
"""FP32 GAIC MobileNetV2 export model with the reference RoI/RoD sampling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2


class FeatureExtractor(nn.Module):
    """Shared MobileNetV2 feature pyramid used by GAIC's reddim16 model."""

    def __init__(self) -> None:
        super().__init__()
        network = mobilenet_v2(weights=None)
        self.feature3 = nn.Sequential(network.features[:7])
        self.feature4 = nn.Sequential(network.features[7:14])
        self.feature5 = nn.Sequential(network.features[14:-1])

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature3 = self.feature3(image)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)
        return feature3, feature4, feature5


def align_features(
    features: torch.Tensor, boxes: torch.Tensor, *, region_difference: bool = False
) -> torch.Tensor:
    """Sample the upstream 10×10 lattice and average to its 9×9 RoI/RoD grid."""
    height, width = features.shape[-2:]
    positions = torch.arange(10, dtype=features.dtype, device=features.device)
    scaled_boxes = boxes * (1.0 / 16.0)
    if region_difference:
        ys = positions[None, :, None] * ((height - 1.001) / 9.0)
        xs = positions[None, None, :] * ((width - 1.001) / 9.0)
        ys = ys.expand(-1, 10, 10)
        xs = xs.expand(-1, 10, 10)
        valid = ~(
            (ys >= scaled_boxes[:, 1, None, None])
            & (ys <= scaled_boxes[:, 3, None, None])
            & (xs >= scaled_boxes[:, 0, None, None])
            & (xs <= scaled_boxes[:, 2, None, None])
        )
    else:
        y_extent = torch.clamp(scaled_boxes[:, 3] - scaled_boxes[:, 1] + 1.0, min=0.0)
        x_extent = torch.clamp(scaled_boxes[:, 2] - scaled_boxes[:, 0] + 1.0, min=0.0)
        ys = scaled_boxes[:, 1, None, None] + positions[None, :, None] * (
            y_extent[:, None, None] / 9.0
        )
        xs = scaled_boxes[:, 0, None, None] + positions[None, None, :] * (
            x_extent[:, None, None] / 9.0
        )
        ys = ys.expand(-1, 10, 10)
        xs = xs.expand(-1, 10, 10)
        valid = (ys >= 0.0) & (ys < height) & (xs >= 0.0) & (xs < width)

    # Keep the reference's final-two-pixel extrapolation, not border clamping.
    y0 = torch.clamp(torch.floor(ys), min=0, max=height - 2).to(torch.int64)
    x0 = torch.clamp(torch.floor(xs), min=0, max=width - 2).to(torch.int64)
    dy = (ys - y0.to(ys.dtype))[:, None]
    dx = (xs - x0.to(xs.dtype))[:, None]
    flat = features[0].flatten(1)

    def gather(y_index: torch.Tensor, x_index: torch.Tensor) -> torch.Tensor:
        indices = (y_index * width + x_index).reshape(-1)
        return (
            flat[:, indices].reshape(features.shape[1], -1, 10, 10).permute(1, 0, 2, 3)
        )

    sampled = (
        gather(y0, x0) * (1.0 - dy) * (1.0 - dx)
        + gather(y0, x0 + 1) * (1.0 - dy) * dx
        + gather(y0 + 1, x0) * dy * (1.0 - dx)
        + gather(y0 + 1, x0 + 1) * dy * dx
    )
    return F.avg_pool2d(torch.where(valid[:, None], sampled, 0.0), 2, 1)


class GaicMobileNetV2(nn.Module):
    """GAIC MobileNetV2 with 16 reduced dimensions and dynamic crop boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.Feat_ext = FeatureExtractor()
        self.DimRed = nn.Conv2d(448, 16, kernel_size=1)
        self.FC_layers = nn.Sequential(
            nn.Sequential(nn.Conv2d(32, 768, kernel_size=9), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(768, 128, kernel_size=1), nn.ReLU(inplace=True)),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def feature_map(self, image: torch.Tensor) -> torch.Tensor:
        feature3, feature4, feature5 = self.Feat_ext(image)
        merged = torch.cat(
            (
                F.interpolate(
                    feature3,
                    size=feature4.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
                feature4,
                0.5
                * F.interpolate(
                    feature5,
                    size=feature4.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ),
            dim=1,
        )
        return self.DimRed(merged)

    def score_features(
        self, features: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        regions = align_features(features, boxes)
        differences = align_features(features, boxes, region_difference=True)
        return self.FC_layers(torch.cat((regions, differences), dim=1)).reshape(-1)

    def forward(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        return self.score_features(self.feature_map(image), boxes)
