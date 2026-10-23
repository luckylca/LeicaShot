"""Reference-style highlight extraction and depth-layered optical scattering.

The base plate stays in sRGB uint8. Only the isolated highlight surplus is
stored in linear float32; scattering is evaluated one output row band at a
time, so each CoC layer needs only a bounded source halo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, hypot, pi, sqrt
from typing import Any

import cv2
import numpy as np

_LINEAR = np.asarray(
    [
        (v / 255.0 / 12.92)
        if v / 255.0 <= 0.04045
        else (((v / 255.0 + 0.055) / 1.055) ** 2.4)
        for v in range(256)
    ],
    dtype=np.float32,
)
_APERTURES = (0.95, 1.8, 2.0, 2.8, 4.0, 5.6)


@dataclass(frozen=True)
class HighlightSplit:
    """Opaque sRGB base plus nonnegative, linear-RGB highlight radiance."""

    base_rgb: Any
    highlight_linear: Any


def _check(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _smooth(value: Any) -> Any:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _canonical_aperture(aperture: float) -> float:
    if not np.isfinite(aperture):
        raise ValueError("unsupported aperture")
    for candidate in _APERTURES:
        if np.float32(aperture) == np.float32(candidate):
            return candidate
    raise ValueError("unsupported aperture")


def _box_mean(image: Any, radius: int, *, horizontal: bool) -> Any:
    """Clipped-window box mean matching the reference's cropped edge windows."""
    height, width = image.shape
    kernel = (2 * radius + 1, 1) if horizontal else (1, 2 * radius + 1)
    summed = cv2.boxFilter(
        image,
        ddepth=cv2.CV_32F,
        ksize=kernel,
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    length = width if horizontal else height
    positions = np.arange(length, dtype=np.int32)
    counts = (
        np.minimum(length - 1, positions + radius)
        - np.maximum(0, positions - radius)
        + 1
    )
    if horizontal:
        summed /= counts.astype(np.float32)[None, :]
    else:
        summed /= counts.astype(np.float32)[:, None]
    return summed


def extract_highlights(
    rgb: Any,
    *,
    cancel_check: Callable[[], None] | None = None,
    row_chunk_height: int = 128,
) -> HighlightSplit:
    """Split a photo as the reference does, retaining diffuse bright surfaces.

    The local-contrast map, connected-component rejection, compact-source
    confidence, RGB linearization and byte quantization follow
    ``HighlightExtraction.caps`` and ``LegendaryMath.split``. Bright regions
    wider than eight percent of the short side are kept on the base plate.
    """
    image = np.asarray(rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
        or not image.flags.c_contiguous
    ):
        raise ValueError("rgb must be contiguous HWC uint8 RGB")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")

    height, width = image.shape[:2]
    count = height * width
    peak = np.empty((height, width), dtype=np.float32)
    for channel in range(3):
        _check(cancel_check)
        values = _LINEAR[image[:, :, channel]]
        if channel == 0:
            peak[:] = values
        else:
            np.maximum(peak, values, out=peak)
        del values

    radius = max(4, int(np.rint(min(width, height) * 0.015)))
    horizontal = _box_mean(peak, radius, horizontal=True)
    local_mean = _box_mean(horizontal, radius, horizontal=False)
    strength = _smooth((peak - np.float32(0.08)) / np.float32(0.12))
    contrast = _smooth((peak - local_mean - np.float32(0.025)) / np.float32(0.10))
    strength *= contrast
    candidate = strength > 0.0
    caps = np.maximum(np.float32(0.25), peak)
    caps[candidate] = np.float32(0.25) - np.float32(0.19) * strength[candidate]
    del horizontal, local_mean, contrast, strength

    _check(cancel_check)
    candidate |= peak > np.float32(0.25)
    if bool(candidate.any()):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8, copy=False), connectivity=8, ltype=cv2.CV_32S
        )
        max_span = max(8, int(np.rint(min(width, height) * 0.08)))
        for label in range(1, component_count):
            _check(cancel_check)
            x0, y0, span_x, span_y, _ = (int(v) for v in stats[label])
            if span_x > max_span or span_y > max_span:
                region_peak = peak[y0 : y0 + span_y, x0 : x0 + span_x]
                caps[y0 : y0 + span_y, x0 : x0 + span_x] = np.maximum(
                    np.float32(0.25), region_peak
                )
                continue

            region_labels = labels[y0 : y0 + span_y, x0 : x0 + span_x]
            component = region_labels == label
            region_peak = peak[y0 : y0 + span_y, x0 : x0 + span_x]
            peak_index = int(np.argmax(np.where(component, region_peak, -1.0)))
            peak_y, peak_x = divmod(peak_index, span_x)
            top_peak = float(region_peak[peak_y, peak_x])
            pixel = image[y0 + peak_y, x0 + peak_x]
            highest = int(pixel.max())
            lowest = int(pixel.min())
            chroma = (highest - lowest) / max(1, highest)

            pad = max(2, min(6, min(span_x, span_y)))
            outer_x0, outer_y0 = max(0, x0 - pad), max(0, y0 - pad)
            outer_x1 = min(width, x0 + span_x + pad)
            outer_y1 = min(height, y0 + span_y + pad)
            ring = peak[outer_y0:outer_y1, outer_x0:outer_x1]
            total = float(ring.sum(dtype=np.float64))
            inner = peak[y0 : y0 + span_y, x0 : x0 + span_x]
            total -= float(inner.sum(dtype=np.float64))
            ring_count = ring.size - span_x * span_y
            surround = total / ring_count if ring_count else top_peak

            neutral = float(_smooth((top_peak - 0.60) / 0.25))
            colored = float(_smooth((chroma - 0.35) / 0.35)) * float(
                _smooth((top_peak - 0.13) / 0.10)
            )
            isolation = float(
                _smooth((1.0 - surround / max(top_peak, 1e-8) - 0.55) / 0.30)
            )
            shape = float(
                _smooth((min(span_x, span_y) / max(span_x, span_y) - 0.34) / 0.20)
            )
            confidence = max(neutral, colored) * isolation * shape
            caps_region = caps[y0 : y0 + span_y, x0 : x0 + span_x]
            component_caps = np.maximum(np.float32(0.25), region_peak)
            if confidence == 0.0:
                caps_region[component] = component_caps[component]
            else:
                reduced = region_peak - (region_peak - caps_region) * np.float32(
                    confidence
                )
                np.maximum(caps_region, reduced, out=caps_region, where=component)

        del labels, stats, candidate

    base = np.empty_like(image)
    high = np.empty(image.shape, dtype=np.float32)
    for row_start in range(0, height, row_chunk_height):
        _check(cancel_check)
        row_stop = min(height, row_start + row_chunk_height)
        region = image[row_start:row_stop]
        region_peak = peak[row_start:row_stop]
        ratio = np.empty_like(region_peak)
        np.divide(
            caps[row_start:row_stop],
            np.maximum(region_peak, np.float32(1e-8)),
            out=ratio,
        )
        np.minimum(ratio, np.float32(1.0), out=ratio)
        for channel in range(3):
            linear = _LINEAR[region[:, :, channel]]
            linear *= ratio
            linear_to_srgb = np.empty_like(linear)
            low = linear <= np.float32(0.0031308)
            linear_to_srgb[low] = np.float32(12.92) * linear[low]
            linear_to_srgb[~low] = np.float32(1.055) * np.power(
                linear[~low], np.float32(1.0 / 2.4)
            ) - np.float32(0.055)
            np.rint(np.clip(linear_to_srgb, 0.0, 1.0) * 255.0, out=linear_to_srgb)
            base[row_start:row_stop, :, channel] = linear_to_srgb.astype(np.uint8)
            high[row_start:row_stop, :, channel] = np.maximum(
                np.float32(0.0),
                _LINEAR[region[:, :, channel]]
                - _LINEAR[base[row_start:row_stop, :, channel]],
            )
            del linear, linear_to_srgb
        del ratio

    _check(cancel_check)
    assert peak.size == count
    return HighlightSplit(base, high)


def aperture_kernel(radius: int, aperture: float) -> Any:
    """Return the reference's normalized, 8×8 sampled nine-blade PSF."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    aperture = _canonical_aperture(aperture)
    if radius == 0:
        return np.ones((1, 1), dtype=np.float32)

    roundness = {0.95: 1.0, 1.8: 0.85, 2.0: 0.8, 2.8: 0.65, 4.0: 0.45, 5.6: 0.25}[
        aperture
    ]
    half = radius + 1
    axis = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(axis, axis)
    distance = np.hypot(dx, dy)
    inner = radius * (roundness + (1.0 - roundness) * cos(pi / 9.0))
    allowance = sqrt(0.5)
    result = np.zeros(distance.shape, dtype=np.float32)
    result[distance < inner - allowance] = 1.0
    edge = (distance >= inner - allowance) & (distance <= radius + allowance)
    if bool(edge.any()):
        edge_y, edge_x = np.nonzero(edge)
        edge_dx = dx[edge_y, edge_x]
        edge_dy = dy[edge_y, edge_x]
        coverage = np.zeros(edge_x.shape, dtype=np.uint8)
        for sub_y in range(8):
            for sub_x in range(8):
                sample_x = edge_dx + (sub_x + 0.5) / 8.0 - 0.5
                sample_y = edge_dy + (sub_y + 0.5) / 8.0 - 0.5
                angle = np.arctan2(sample_y, sample_x)
                period = 2.0 * pi / 9.0
                sector = (angle + pi / 9.0) % period - pi / 9.0
                boundary = radius * (
                    roundness + (1.0 - roundness) * cos(pi / 9.0) / np.cos(sector)
                )
                coverage += (np.hypot(sample_x, sample_y) <= boundary).astype(np.uint8)
        result[edge_y, edge_x] = coverage.astype(np.float32) / np.float32(64.0)
    total = float(result.sum(dtype=np.float64))
    if total <= 0.0:
        raise RuntimeError("aperture PSF has zero coverage")
    result /= np.float32(total)
    return result


def _optical_pupil_kernel(
    radius: int, aperture: float, ux: float, uy: float, radial_ratio: float
) -> Any:
    """Reference field-dependent pupil kernel for one spatial grid node."""
    if radius == 0 or radial_ratio == 1.0:
        return aperture_kernel(radius, aperture).astype(np.float64)
    roundness = {0.95: 1.0, 1.8: 0.85, 2.0: 0.8, 2.8: 0.65, 4.0: 0.45, 5.6: 0.25}[
        aperture
    ]
    half = radius + 1
    axis = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(axis, axis)
    extra = 1.0 / radial_ratio - 1.0
    projected = (dx * ux + dy * uy) * extra
    warped_x = dx + projected * ux
    warped_y = dy + projected * uy
    distance = np.hypot(warped_x, warped_y)
    inner = radius * (roundness + (1.0 - roundness) * cos(pi / 9.0))
    allowance = sqrt(0.5) / radial_ratio
    result = np.zeros(distance.shape, dtype=np.float32)
    result[distance < inner - allowance] = 1.0
    edge = (distance >= inner - allowance) & (distance <= radius + allowance)
    if bool(edge.any()):
        edge_y, edge_x = np.nonzero(edge)
        edge_dx = dx[edge_y, edge_x]
        edge_dy = dy[edge_y, edge_x]
        coverage = np.zeros(edge_x.shape, dtype=np.uint8)
        for sub_y in range(8):
            for sub_x in range(8):
                sample_x = edge_dx + (sub_x + 0.5) / 8.0 - 0.5
                sample_y = edge_dy + (sub_y + 0.5) / 8.0 - 0.5
                sample_projection = (sample_x * ux + sample_y * uy) * extra
                sample_x += sample_projection * ux
                sample_y += sample_projection * uy
                angle = np.arctan2(sample_y, sample_x)
                period = 2.0 * pi / 9.0
                sector = (angle + pi / 9.0) % period - pi / 9.0
                boundary = radius * (
                    roundness + (1.0 - roundness) * cos(pi / 9.0) / np.cos(sector)
                )
                coverage += (np.hypot(sample_x, sample_y) <= boundary).astype(np.uint8)
        result[edge_y, edge_x] = coverage.astype(np.float32) / np.float32(64.0)
    total = float(result.sum(dtype=np.float64))
    if total <= 0.0:
        raise RuntimeError("optical pupil PSF has zero coverage")
    result /= np.float32(total)
    return result.astype(np.float64)


def _canonical_pupil_shape(
    shape: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Canonicalize opposite field axes, whose projected pupil is identical."""
    ux, uy, radial_ratio = shape
    if ux < 0.0 or (ux == 0.0 and uy < 0.0):
        return -ux, -uy, radial_ratio
    return ux, uy, radial_ratio


def _field_pupil_kernels(
    shapes: list[tuple[float, float, float]], radius: int, aperture: float
) -> list[Any]:
    """Build each distinct field pupil once and reuse it for opposite nodes."""
    cache: dict[tuple[float, float, float], Any] = {}
    kernels = []
    for shape in shapes:
        key = _canonical_pupil_shape(shape)
        if key not in cache:
            cache[key] = _optical_pupil_kernel(radius, aperture, *key)
        kernels.append(cache[key])
    return kernels


def _field_axis_nodes(size: int) -> list[tuple[float, Any, int, int]]:
    if size == 1:
        return [(0.5, np.ones(1, dtype=np.float32), 0, 1)]
    positions = np.arange(size, dtype=np.float64) / (size - 1) * 4.0
    nodes = []
    for index in range(5):
        weights = np.maximum(0.0, 1.0 - np.abs(positions - index)).astype(np.float32)
        support = np.flatnonzero(weights > 0.0)
        nodes.append((index / 4.0, weights, int(support[0]), int(support[-1]) + 1))
    return nodes


def _optical_shape(
    x: float, y: float, width: int, height: int
) -> tuple[float, float, float]:
    dx = (x - 0.5) * (width - 1)
    dy = (y - 0.5) * (height - 1)
    distance = hypot(dx, dy)
    bound = hypot((width - 1) / 2.0, (height - 1) / 2.0)
    t = min(1.0, max(0.0, distance / bound)) if bound else 0.0
    smooth = t * t * (3.0 - 2.0 * t)
    ratio = 1.0 - (1.0 - 0.82) * smooth
    if distance == 0.0:
        return 1.0, 0.0, ratio
    return dx / distance, dy / distance, ratio


def _reflected_rows(start: int, stop: int, halo: int, height: int) -> Any:
    raw = np.arange(start - halo, stop + halo, dtype=np.int64)
    if height == 1:
        return np.zeros(raw.shape, dtype=np.intp)
    period = 2 * height - 2
    mod = np.mod(raw, period)
    return np.asarray(np.where(mod < height, mod, period - mod), dtype=np.intp)


def _fft32_mass_filter_interior(
    source: Any,
    kernel: Any,
    target_height: int,
    target_width: int,
    kernel_spectrum_cache: dict[tuple[int, int, int], Any],
) -> Any:
    """FFT-convolve a halo-padded mass tile with a cached FP32 pupil spectrum."""
    dft_height = cv2.getOptimalDFTSize(source.shape[0] + kernel.shape[0] - 1)
    dft_width = cv2.getOptimalDFTSize(source.shape[1] + kernel.shape[1] - 1)
    cache_key = (id(kernel), dft_height, dft_width)
    kernel_spectrum = kernel_spectrum_cache.get(cache_key)
    if kernel_spectrum is None:
        kernel_pad = np.zeros((dft_height, dft_width), dtype=np.float32)
        kernel_pad[: kernel.shape[0], : kernel.shape[1]] = kernel[::-1, ::-1]
        kernel_spectrum = cv2.dft(kernel_pad, flags=cv2.DFT_COMPLEX_OUTPUT)
        kernel_spectrum_cache[cache_key] = kernel_spectrum
        del kernel_pad

    source_pad = np.zeros((dft_height, dft_width), dtype=np.float32)
    source_pad[: source.shape[0], : source.shape[1]] = source
    source_spectrum = cv2.dft(source_pad, flags=cv2.DFT_COMPLEX_OUTPUT)
    product = cv2.mulSpectrums(source_spectrum, kernel_spectrum, 0)
    del source_pad, source_spectrum, kernel_spectrum
    full = cv2.idft(product, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    del product

    kernel_half_y = kernel.shape[0] // 2
    kernel_half_x = kernel.shape[1] // 2
    start_y = 2 * kernel_half_y
    start_x = 2 * kernel_half_x
    return full[
        start_y : start_y + target_height,
        start_x : start_x + target_width,
    ]


def _source_levels(
    coc: Any, scale: float, cancel_check: Callable[[], None] | None
) -> list[int]:
    """Return every present CoC layer, including empty-light occluders."""
    height = coc.shape[0]
    levels: set[int] = set()
    for y0 in range(0, height, 128):
        _check(cancel_check)
        y1 = min(height, y0 + 128)
        values = coc[y0:y1] * np.float32(scale)
        levels.update(np.unique(np.floor(values).astype(np.int32)).tolist())
        levels.update(np.unique(np.ceil(values).astype(np.int32)).tolist())
    return sorted(levels, reverse=True)


def _scatter_field_optical(
    high: Any,
    depth: Any,
    aperture: float,
    radius_scale: float,
    row_chunk_height: int,
    levels: list[int],
    cancel_check: Callable[[], None] | None,
    progress_callback: Callable[[float], None] | None,
    *,
    skip_empty_mass: bool = True,
    mass_filter_float32: bool = False,
    mass_filter_fft32_outer: bool = False,
) -> Any:
    """Reference 5×5 tent-interpolated pupil, evaluated through bounded tiles."""
    height, width = depth.shape
    output = np.zeros_like(high)
    own = np.zeros_like(high)
    coverage = np.zeros((height, width), dtype=np.float32)
    xs = _field_axis_nodes(width)
    ys = _field_axis_nodes(height)
    nodes: list[tuple[Any, ...]] = []
    for y, wy, y0, y1 in ys:
        for x, wx, x0, x1 in xs:
            nodes.append(
                (x, y, wx, wy, x0, x1, y0, y1, _optical_shape(x, y, width, height))
            )
    pupil_shapes = [node[8] for node in nodes]

    for level_index, level in enumerate(levels):
        _check(cancel_check)
        radius = abs(level)
        if radius == 0:
            zero_kernel = aperture_kernel(0, aperture).astype(np.float64)
            kernels = [zero_kernel] * len(nodes)
        else:
            kernels = _field_pupil_kernels(pupil_shapes, radius, aperture)
        if mass_filter_float32:
            mass_kernel_cache: dict[int, Any] = {}
            mass_kernels = []
            for kernel in kernels:
                kernel_id = id(kernel)
                if kernel_id not in mass_kernel_cache:
                    mass_kernel_cache[kernel_id] = kernel.astype(np.float32)
                mass_kernels.append(mass_kernel_cache[kernel_id])
        else:
            mass_kernels = kernels
        fft_kernel_spectrum_cache: dict[tuple[int, int, int], Any] = {}
        for out_y0 in range(0, height, row_chunk_height):
            _check(cancel_check)
            out_y1 = min(height, out_y0 + row_chunk_height)
            row_count = out_y1 - out_y0
            mass_total = np.zeros((row_count, width), dtype=np.float64)
            radiance_total = np.zeros((row_count, width, 3), dtype=np.float64)

            for node, kernel, mass_kernel in zip(
                nodes, kernels, mass_kernels, strict=True
            ):
                _, _, wx, wy, x0, x1, node_y0, node_y1, _ = node
                y0 = max(out_y0, node_y0)
                y1 = min(out_y1, node_y1)
                if y0 >= y1 or not bool(np.any(wy[y0:y1] > 0.0)):
                    continue
                half = kernel.shape[0] // 2
                source_y = _reflected_rows(y0, y1, half, height)
                source_x = _reflected_rows(x0, x1, half, width)
                selection = np.ix_(source_y, source_x)
                source_coc = depth[selection] * np.float32(radius_scale)
                layer_weight = np.maximum(
                    np.float32(0.0),
                    np.float32(1.0) - np.abs(source_coc - np.float32(level)),
                )
                if skip_empty_mass and not bool(np.any(layer_weight)):
                    del selection, source_coc, layer_weight
                    continue
                source_high = high[selection]
                weighted = None
                if np.any(source_high):
                    weighted = np.empty(source_high.shape, dtype=np.float64)
                    np.multiply(source_high, layer_weight[:, :, None], out=weighted)
                use_fft32 = mass_filter_fft32_outer and layer_weight.shape[1] <= 1700
                if use_fft32:
                    node_mass = _fft32_mass_filter_interior(
                        layer_weight,
                        kernel,
                        y1 - y0,
                        x1 - x0,
                        fft_kernel_spectrum_cache,
                    )
                    mass = None
                else:
                    mass = cv2.filter2D(
                        layer_weight
                        if mass_filter_float32
                        else layer_weight.astype(np.float64),
                        ddepth=cv2.CV_32F if mass_filter_float32 else cv2.CV_64F,
                        kernel=mass_kernel,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    node_mass = mass[half : half + y1 - y0, half : half + x1 - x0]
                node_weight = (
                    wy[y0:y1].astype(np.float64)[:, None]
                    * wx[x0:x1].astype(np.float64)[None, :]
                )
                row_offset = y0 - out_y0
                rows = slice(row_offset, row_offset + y1 - y0)
                mass_total[rows, x0:x1] += node_mass * node_weight
                # Keep the mass filter for occlusion; only empty radiance is skipped.
                if weighted is not None and np.any(weighted):
                    radiance = cv2.filter2D(
                        weighted,
                        ddepth=cv2.CV_64F,
                        kernel=kernel,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    node_radiance = radiance[
                        half : half + y1 - y0, half : half + x1 - x0
                    ]
                    radiance_total[rows, x0:x1] += (
                        node_radiance * node_weight[:, :, None]
                    )
                del (
                    selection,
                    source_high,
                    source_coc,
                    layer_weight,
                    weighted,
                    mass,
                )
            receiver_coc = depth[out_y0:out_y1] * np.float32(radius_scale)
            alpha = np.clip(mass_total, 0.0, 1.0).astype(np.float32)
            behind = np.clip(receiver_coc - np.float32(level + 1), 0.0, 1.0)
            footprint = np.minimum(
                1.0,
                np.abs(receiver_coc) / np.float32(max(abs(level), 1)),
            )
            visibility = 1.0 - behind * (1.0 - footprint)
            take = np.minimum(
                alpha * visibility,
                np.maximum(0.0, 1.0 - coverage[out_y0:out_y1]),
            )
            ratio = np.zeros_like(alpha)
            np.divide(take, alpha, out=ratio, where=alpha > np.float32(1e-8))
            local = np.maximum(0.0, 1.0 - np.abs(receiver_coc - np.float32(level)))
            own_ratio = np.zeros_like(alpha)
            np.divide(local, alpha, out=own_ratio, where=alpha > np.float32(1e-8))
            radiance32 = np.maximum(0.0, radiance_total).astype(np.float32)
            output[out_y0:out_y1] += radiance32 * ratio[:, :, None]
            own[out_y0:out_y1] += radiance32 * own_ratio[:, :, None]
            coverage[out_y0:out_y1] += take
            del mass_total, radiance_total, receiver_coc, alpha, behind, footprint
            del visibility, take, ratio, local, own_ratio, radiance32
        del fft_kernel_spectrum_cache
        if progress_callback is not None:
            progress_callback((level_index + 1) / len(levels))

    _combine_own_into_output(output, own, coverage, row_chunk_height)
    _check(cancel_check)
    return output


def _combine_own_into_output(
    output: Any, own: Any, coverage: Any, row_chunk_height: int
) -> None:
    """Apply the residual local light without a second full-frame temporary."""
    height = coverage.shape[0]
    for y0 in range(0, height, row_chunk_height):
        y1 = min(height, y0 + row_chunk_height)
        residual_coverage = 1.0 - coverage[y0:y1]
        output[y0:y1] += own[y0:y1] * residual_coverage[:, :, None]


def scatter_highlights(
    highlight_linear: Any,
    coc: Any,
    aperture: float,
    *,
    radius_scale: float = 3.0,
    field_optics: bool = False,
    row_chunk_height: int = 64,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> Any:
    """Scatter linear highlight radiance through the aperture PSF.

    Integer CoC layers use a tent partition of unity. Layers are accumulated
    near-to-far using the same saturated coverage and relative-depth
    visibility rule as the Kotlin compositor. `radius_scale=3` matches the
    production highlight enlargement; `field_optics=True` applies the spatial
    5×5 pupil grid. The CoC used by ARNet remains intact.
    """
    high = np.asarray(highlight_linear)
    depth = np.asarray(coc)
    if (
        high.dtype != np.float32
        or high.ndim != 3
        or high.shape[2] != 3
        or not high.flags.c_contiguous
        or depth.dtype != np.float32
        or depth.ndim != 2
        or depth.shape != high.shape[:2]
        or not depth.flags.c_contiguous
    ):
        raise ValueError(
            "high must be contiguous HWC float32 and coc matching HW float32"
        )
    aperture = _canonical_aperture(aperture)
    if not np.isfinite(radius_scale) or not 1.0 <= radius_scale <= 3.0:
        raise ValueError("radius_scale must be in [1, 3]")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    height, width = depth.shape
    for y0 in range(0, height, 128):
        _check(cancel_check)
        high_region = high[y0 : y0 + 128]
        coc_region = depth[y0 : y0 + 128]
        if not bool(np.isfinite(high_region).all()) or float(high_region.min()) < 0.0:
            raise ValueError("highlight radiance must be finite and nonnegative")
        if (
            not bool(np.isfinite(coc_region).all())
            or float(np.abs(coc_region).max()) > 54.0
        ):
            raise ValueError("CoC must be finite and within the reference bound ±54")

    levels = _source_levels(depth, radius_scale, cancel_check)
    if not levels:
        return np.zeros_like(high)
    if field_optics:
        return _scatter_field_optical(
            high,
            depth,
            aperture,
            radius_scale,
            row_chunk_height,
            levels,
            cancel_check,
            progress_callback,
        )

    output = np.zeros_like(high)
    coverage = np.zeros((height, width), dtype=np.float32)
    own = np.zeros_like(high)
    kernel_cache: dict[int, Any] = {}
    radius_cache: dict[int, int] = {}
    for level_index, level in enumerate(levels):
        _check(cancel_check)
        radius = abs(level)
        if radius not in kernel_cache:
            kernel_cache[radius] = aperture_kernel(radius, aperture).astype(np.float64)
            radius_cache[radius] = radius + 1 if radius else 0
        kernel = kernel_cache[radius]
        halo = radius_cache[radius]
        for y0 in range(0, height, row_chunk_height):
            _check(cancel_check)
            y1 = min(height, y0 + row_chunk_height)
            source_rows = _reflected_rows(y0, y1, halo, height)
            source_high = high[source_rows]
            scaled_coc = depth[source_rows] * np.float32(radius_scale)
            weights = np.maximum(
                np.float32(0.0), np.float32(1.0) - np.abs(scaled_coc - level)
            )
            weighted = np.empty(source_high.shape, dtype=np.float64)
            np.multiply(source_high, weights[:, :, None], out=weighted)
            mass = cv2.filter2D(
                weights.astype(np.float64),
                ddepth=cv2.CV_64F,
                kernel=kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
            radiance = cv2.filter2D(
                weighted,
                ddepth=cv2.CV_64F,
                kernel=kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
            begin = halo
            end = begin + (y1 - y0)
            alpha = np.clip(mass[begin:end], 0.0, 1.0)
            receiver_coc = scaled_coc[begin:end]
            behind = np.clip(receiver_coc - np.float32(level + 1), 0.0, 1.0)
            footprint = np.minimum(
                1.0,
                np.abs(receiver_coc) / np.float32(max(abs(level), 1)),
            )
            visibility = 1.0 - behind * (1.0 - footprint)
            take = np.minimum(
                alpha * visibility, np.maximum(0.0, 1.0 - coverage[y0:y1])
            )
            ratio = np.zeros_like(alpha)
            np.divide(take, alpha, out=ratio, where=alpha > np.float32(1e-8))
            local = np.maximum(0.0, 1.0 - np.abs(receiver_coc - np.float32(level)))
            own_ratio = np.zeros_like(alpha)
            np.divide(local, alpha, out=own_ratio, where=alpha > np.float32(1e-8))
            filtered = radiance[begin:end]
            output[y0:y1] += filtered * ratio[:, :, None]
            own[y0:y1] += filtered * own_ratio[:, :, None]
            coverage[y0:y1] += take
            del source_high, scaled_coc, weights, weighted, mass, radiance
        if progress_callback is not None:
            progress_callback((level_index + 1) / len(levels))

    _combine_own_into_output(output, own, coverage, row_chunk_height)
    _check(cancel_check)
    return output


def recover_clipped_highlights_inplace(
    highlight_linear: Any,
    source_rgb: Any,
    coc: Any,
    *,
    row_chunk_height: int = 128,
    cancel_check: Callable[[], None] | None = None,
) -> bool:
    """Recover clipped, defocused point-light radiance using coarse local density."""
    high = np.asarray(highlight_linear)
    source = np.asarray(source_rgb)
    depth = np.asarray(coc)
    if (
        high.dtype != np.float32
        or high.ndim != 3
        or high.shape[2] != 3
        or not high.flags.c_contiguous
        or source.dtype != np.uint8
        or source.shape != high.shape
        or depth.dtype != np.float32
        or depth.shape != high.shape[:2]
        or not depth.flags.c_contiguous
    ):
        raise ValueError("highlight, source RGB, and CoC buffers must match")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")
    height, width = depth.shape

    peak = np.maximum(high[:, :, 0], high[:, :, 1])
    np.maximum(peak, high[:, :, 2], out=peak)
    pad_h = (-height) % 4
    pad_w = (-width) % 4
    padded = np.pad(peak, ((0, pad_h), (0, pad_w)))
    cell = padded.reshape((padded.shape[0] // 4, 4, padded.shape[1] // 4, 4)).sum(
        axis=(1, 3), dtype=np.float64
    )
    del peak, padded
    grid_h, grid_w = cell.shape
    summed = np.zeros((grid_h + 1, grid_w + 1), dtype=np.float64)
    summed[1:, 1:] = cell.cumsum(axis=0).cumsum(axis=1)
    del cell
    x = np.arange(width, dtype=np.int32)[None, :]
    changed = False

    for y0 in range(0, height, row_chunk_height):
        _check(cancel_check)
        y1 = min(height, y0 + row_chunk_height)
        source_chunk = source[y0:y1]
        depth_chunk = depth[y0:y1]
        high_chunk = high[y0:y1]
        max_byte = source_chunk.max(axis=2)
        peak_high = np.maximum(high_chunk[:, :, 0], high_chunk[:, :, 1])
        np.maximum(peak_high, high_chunk[:, :, 2], out=peak_high)

        clip_weight = _smooth(
            (max_byte.astype(np.float32) - np.float32(236.0)) / np.float32(18.0)
        )
        depth_weight = _smooth(
            (np.abs(depth_chunk) - np.float32(0.75)) / np.float32(1.25)
        )
        tone_weight = _smooth(
            (
                peak_high / np.maximum(_LINEAR[max_byte], np.float32(1e-8))
                - np.float32(0.25)
            )
            / np.float32(0.5)
        )
        radius = np.maximum(np.float32(1.0), np.abs(depth_chunk) * np.float32(3.0))
        radius_int = np.ceil(radius).astype(np.int32)
        y = np.arange(y0, y1, dtype=np.int32)[:, None]
        x0 = np.maximum(0, x - radius_int) // 4
        y0_cells = np.maximum(0, y - radius_int) // 4
        x1 = np.minimum(width - 1, x + radius_int) // 4 + 1
        y1_cells = np.minimum(height - 1, y + radius_int) // 4 + 1
        x0 = np.minimum(x0, grid_w - 1)
        x1 = np.minimum(x1, grid_w)
        y0_cells = np.minimum(y0_cells, grid_h - 1)
        y1_cells = np.minimum(y1_cells, grid_h)
        local_sum = (
            summed[y1_cells, x1]
            - summed[y0_cells, x1]
            - summed[y1_cells, x0]
            + summed[y0_cells, x0]
        )
        mean = local_sum / (pi * radius.astype(np.float64) ** 2)
        wanted = np.ones(mean.shape, dtype=np.float32)
        np.divide(np.float32(0.5), mean, out=wanted, where=mean > 0.0)
        np.clip(wanted, 1.0, 48.0, out=wanted)
        gain = (
            np.float32(1.0)
            + (wanted - np.float32(1.0)) * clip_weight * depth_weight * tone_weight
        )
        eligible = (max_byte > 236) & (np.abs(depth_chunk) > np.float32(0.75))
        gain[~eligible] = np.float32(1.0)
        if bool(np.any(gain > 1.0)):
            high_chunk *= gain[:, :, None]
            changed = True
        del max_byte, peak_high, clip_weight, depth_weight, tone_weight
        del (
            radius,
            radius_int,
            x0,
            y0_cells,
            x1,
            y1_cells,
            local_sum,
            mean,
            wanted,
            gain,
        )

    _check(cancel_check)
    return changed


def soften_highlights_inplace(
    light: Any,
    coc: Any,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> bool:
    """Apply the reference's CoC-weighted, separable tent softening in place."""
    frame = np.asarray(light)
    depth = np.asarray(coc)
    if (
        frame.dtype != np.float32
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not frame.flags.c_contiguous
        or depth.dtype != np.float32
        or depth.shape != frame.shape[:2]
        or not depth.flags.c_contiguous
    ):
        raise ValueError("light must be contiguous HWC float32 with matching CoC")
    _check(cancel_check)
    half = np.abs(depth) * np.float32(0.24)
    weight = _smooth((half - np.float32(0.5)) / np.float32(0.5))
    if not bool(np.any(weight > 0.0)):
        return False
    height, width = depth.shape

    def soften_axis(values: Any, axis_weight: Any, axis_half: Any) -> None:
        line_count, line_length = values.shape[:2]
        positions = np.arange(line_length, dtype=np.float64)
        for line_index in range(line_count):
            if line_index % 16 == 0:
                _check(cancel_check)
            row_weight = axis_weight[line_index].astype(np.float64, copy=False)
            if not bool(np.any(row_weight > 0.0)):
                continue
            row = values[line_index]
            half_width = axis_half[line_index].astype(np.float64, copy=False)
            weighted_color = row.astype(np.float64) * row_weight[:, None]
            prefix_weight = np.empty(line_length + 1, dtype=np.float64)
            prefix_color = np.empty((line_length + 1, 3), dtype=np.float64)
            prefix_weight[0] = 0.0
            prefix_color[0] = 0.0
            np.cumsum(row_weight, out=prefix_weight[1:])
            np.cumsum(weighted_color, axis=0, out=prefix_color[1:])

            def integral(points: Any, prefix: Any, cells: Any) -> Any:
                shifted = np.clip(points + 0.5, 0.0, float(line_length))
                index = np.floor(shifted).astype(np.intp)
                fraction = shifted - index
                safe_index = np.minimum(index, line_length - 1)
                if prefix.ndim == 1:
                    result = prefix[index].copy()
                    valid = index < line_length
                    result[valid] += fraction[valid] * cells[safe_index[valid]]
                    return result
                result = prefix[index].copy()
                valid = index < line_length
                result[valid] += fraction[valid, None] * cells[safe_index[valid]]
                return result

            low = positions - half_width
            high = positions + half_width
            mass = integral(high, prefix_weight, row_weight) - integral(
                low, prefix_weight, row_weight
            )
            color = integral(high, prefix_color, weighted_color) - integral(
                low, prefix_color, weighted_color
            )
            selected = (row_weight > 0.0) & (mass > 1e-12)
            if bool(selected.any()):
                average = (color[selected] / mass[selected, None]).astype(np.float32)
                # The axis view may be transposed, so use the row's own weights.
                amount = row_weight[selected].astype(np.float32)[:, None]
                row[selected] = np.maximum(
                    0.0, row[selected] + amount * (average - row[selected])
                )
            del weighted_color, prefix_weight, prefix_color

    horizontal = frame
    horizontal_weight = weight
    horizontal_half = half
    vertical = frame.transpose(1, 0, 2)
    vertical_weight = weight.T
    vertical_half = half.T
    for _ in range(2):
        soften_axis(horizontal, horizontal_weight, horizontal_half)
        _check(cancel_check)
        soften_axis(vertical, vertical_weight, vertical_half)
        _check(cancel_check)
    return True


def composite_highlights(
    base_rgb: Any,
    rendered_rgb: Any,
    scattered_linear: Any,
    *,
    row_chunk_height: int = 128,
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Add scattered light in linear RGB with the reference highlight shoulder."""
    base = np.asarray(base_rgb)
    rendered = np.asarray(rendered_rgb)
    light = np.asarray(scattered_linear)
    if (
        base.dtype != np.uint8
        or base.ndim != 3
        or base.shape[2] != 3
        or rendered.dtype != np.uint8
        or rendered.shape != base.shape
        or light.dtype != np.float32
        or light.shape != base.shape
    ):
        raise ValueError("base, rendered, and scattered light shapes/types must match")
    if row_chunk_height < 1:
        raise ValueError("row_chunk_height must be positive")

    result = np.empty_like(rendered)
    height = base.shape[0]
    for y0 in range(0, height, row_chunk_height):
        _check(cancel_check)
        y1 = min(height, y0 + row_chunk_height)
        source = rendered[y0:y1]
        source_linear = _LINEAR[source]
        amount = light[y0:y1]
        total = source_linear + amount
        result_linear = total.copy()
        compress = (amount > 0.0) & (total > np.float32(0.8)) & (source_linear < 1.0)
        start = np.maximum(source_linear, np.float32(0.8))
        room = 1.0 - start
        rest = amount - (start - source_linear)
        selected = compress & (room > 0.0)
        result_linear[selected] = start[selected] + room[selected] * (
            1.0 - np.exp(-rest[selected] / room[selected])
        )
        np.clip(result_linear, 0.0, 1.0, out=result_linear)
        encoded = np.empty_like(result_linear)
        low = result_linear <= np.float32(0.0031308)
        encoded[low] = np.float32(12.92) * result_linear[low]
        encoded[~low] = np.float32(1.055) * np.power(
            result_linear[~low], np.float32(1.0 / 2.4)
        ) - np.float32(0.055)
        result[y0:y1] = np.rint(encoded * np.float32(255.0)).astype(np.uint8)
    return result
