# Lab color fusion and row streaming

## Algorithm boundary

The inspected reference converts original and predicted RGB bytes to OpenCV
8-bit Lab, keeps high-frequency structure from the original, and transfers the
predicted low-frequency difference independently per channel:

```text
result_c = clip(original_lab_c + Gaussian(predicted_lab_c - original_lab_c))
```

It uses sigma 25 for L, sigma 20 for a/b, OpenCV's reflect-101 border, then
converts the fused 8-bit Lab image back to RGB. `laika.color.fuse_lab_detail`
implements this math using OpenCV native operations. It does not include a
color prediction model and does not load M9 or any other weights.

To reduce temporary memory for tall frames, the function processes row bands
with a vertical halo equal to the largest Gaussian radius (100 pixels at the
default sigma). It preserves the complete requested raster and crops only the
halo from the calculated result. Synthetic tests compare the row-streamed
output byte-for-byte against a full-frame call. An independent two-blur oracle
checks the color formula with the same tolerances used by the reference's
synthetic fusion regression.

## License boundary

The Laika implementation is independently written around OpenCV and NumPy; it
does not copy the reference Kotlin implementation. The reference M9 checkpoint
remains excluded under its personal non-commercial restriction. This stage is
useful only with a separately licensed predictor or caller-provided predicted
RGB. It is not an AI color engine by itself.

See the [row-streaming benchmark](../benchmarks/m3-lab-fusion-row-streaming-2026-10-21.md)
for fresh-worker RSS and time measurements at 2048×1536 and 3840×2160.
