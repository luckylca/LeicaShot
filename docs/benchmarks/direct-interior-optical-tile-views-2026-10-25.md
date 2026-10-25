# In-bounds optical tile views: exact output, universal pixel sizes

## Algorithm (not bound to 20MP)

The original field-dependent 5x5-pupil scatter loop creates NumPy
advanced-index copies of source CoC and source RGB for every optical
node, every depth layer and every output band, even for simple
in-bounds rectangular source windows and entirely dark source tiles.

For an in-bounds source window, the identical source raster can be
addressed directly with ordinary NumPy slices, preserving pixel
coordinates, row order, dtype and kernel halo. Read-only uses of the
source colors need no eager copy; a contiguous copy is created only
when actual source radiance is about to be weighted in-place, so the
caller-owned original input is never mutated.

For tiles crossing an image boundary, the original reflected-indices
advanced indexing is retained exactly. All reflect101 edges, layered
FP32 CoC tent weights, FP64 mass convolution, FP32 RGB radiance
convolution, depth occlusion and layer accumulation are unchanged.
The private _direct_tile_views=False flag restores the old behavior
to support byte-for-byte parity tests.

This is general for any valid image width/height, aperture and CoC.
The gain depends on how much optical work lies inside image borders;
small images, all-edge highlights or shallow pupils may benefit less.

## Windows CPU results: genuine real 5472 x 3648 photo

Hardware Intel Core i9-12900H, Python 3.11, original same authorized
five ONNX checkpoint models, 9 ONNX CPU threads, low-memory ARNet
arena off and the same field optics / output scaling.

| Benchmark | Merged reference | Direct tile views |
|---|---:|---:|
| Optical 20MP isolated (first A/B) | 43.2043 s | 40.0561 s |
| Optical later candidate verification | -- | 39.1781 s |
| Optical full end-to-end instrumented stage | 43.8685 s | 40.5812 s |
| Full 20MP M9+Depth+SAM+ViTMatte+ARNet+optics request | 125.7652 s | 123.4563 s |
| Full-process sampled RSS peak | 1,319,796,736 B | 1,337,913,344 B |

On the first isolated pair the optical stage was ~7.3% faster;
full request was ~1.8% faster across separate runs.
A later trial was ~39.18s for direct views. Run-to-run variance
remains material. Whole-process sampled RSS was ~17.3 MiB higher
in one run and isolated optical RSS was slightly lower; neither
establishes a reliable end-to-end memory decrease.

Both isolated 20MP optical FP32 outputs had EXACT original SHA256:
c0fe387fd6efbd60ed7836d0f44519c28fdf6802401342a2466ebee57c3465ed.

The full 5472x3648 PNG from the actual M9/Depth/SAM/ViTMatte/
ARNet CPU processing request was BYTE IDENTICAL to the existing
develop output; decoded RGB 8-bit maximum error zero.

## 2048px real optical A/B

Captured real 2048x1365 optical input, original field optics,
old/new/new/old fresh process interleaving:

| Mode | Run times (seconds) |
|---|---|
| Original advanced indexing | 2.2153 / 2.1194 |
| Direct source views | 2.1330 / 2.0658 |

These indicate ~3.1% less median optical elapsed time at 2048px,
with EXACT FP32 output SHA256:
a477000897bd26607ffb298f9afdb19d7daef794c49bf4cef72a6ced9bc4821e.

Twelve new parameterized pixel-exact and input-nonmutation tests
exercise no highlights, sparse, border, dense, row chunk sizes
5/17/64, multiple CoC extrema and conservative reflected boundaries.

## Explicitly rejected combinations

- In-place FP32 CoC layer-weight scratch computation was tested
  independently and combined with direct tile views, with no
  reproducible additional speedup. Removed from this deliverable.
- Parallel optical pupil kernel generation was measured slower
  than sequential on the same Windows CPU (2/4/8 workers).
- Moving OpenCV thread count between 1/2/4/8/20 showed no
  stable performance improvement on the 2048px input.
- Pupil subpixel SIMD-style vectorization exactly matched
  1053 reference kernels, but kernel-generation speed only
  improved ~10%, a small part of full native runtime.
  This is held separately, not merged by this PR.
- All named exploratory artifacts and original ONNX weights
  remain under local-data, outside Git.

The original Android project remains a read-only ADB physical
reference; Python remains the formal model engine and backend.
Strict Linux memory cgroups and multiworker service performance
have not been claimed or evaluated in this change.
