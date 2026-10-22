# Third-party notices

## GAIC MobileNetV2 reddim16 implementation

Laika's `tools/gaic_mobile_port.py` adapts the GAIC-Pytorch MobileNetV2,
feature-fusion, crop-sampling, and scoring implementation from
[bo-zhang-cs/GAIC-Pytorch](https://github.com/bo-zhang-cs/GAIC-Pytorch), pinned
to commit `877d6a325943c0a098324dc6d4a975ff4097c3f6`. The upstream code is
copyright (c) 2022 bo-zhang and licensed under MIT. Laika replaces the custom
CUDA RoI/RoD operators with FP32 tensor sampling that preserves the model's
10×10 lattice and 9×9 pooling rule for ONNX CPU inference.

The corresponding model checkpoint is a separate local asset. Its project use
is authorized by the user; no public redistribution rights are asserted. The
checkpoint is not committed to this repository.

MIT License

Copyright (c) 2022 bo-zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
