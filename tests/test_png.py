from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from laika.png import PngRGBStreamWriter


def test_png_stream_writer_preserves_rgb_pixels_and_icc_profile() -> None:
    rng = np.random.default_rng(178)
    expected = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    icc_profile = b"Laika PNG ICC profile fixture"
    stream = BytesIO()
    writer = PngRGBStreamWriter(
        stream, 53, 37, icc_profile=icc_profile, compression_level=2
    )
    writer.write_rows(expected[:11])
    writer.write_rows(expected[11:29])
    writer.write_rows(expected[29:])
    writer.finish()

    stream.seek(0)
    with Image.open(stream) as image:
        image.load()
        actual = np.asarray(image)
        assert image.format == "PNG"
        assert image.size == (53, 37)
        assert image.info["icc_profile"] == icc_profile
    np.testing.assert_array_equal(actual, expected)
