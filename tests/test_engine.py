from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageOps

from laika.engine import ImageInputError, ImageLimitError, ImageLimits, process_image


def _pixels(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (17, 93, 211))


def test_process_image_preserves_raster_and_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "result.png"
    image = _pixels(23, 15)
    image.putpixel((12, 7), (1, 2, 3))
    image.save(source)

    result = process_image(source, destination)

    with Image.open(destination) as output:
        assert output.format == "PNG"
        assert output.size == image.size
        assert output.mode == "RGB"
        assert output.tobytes() == image.tobytes()
    assert result.input_format == "PNG"
    assert (result.output_width, result.output_height) == (23, 15)
    assert set(result.stage_seconds) == {
        "validate",
        "decode",
        "orient_normalize",
        "encode",
    }
    assert all(seconds >= 0 for seconds in result.stage_seconds.values())
    assert result.total_seconds >= 0


def test_process_image_applies_exif_orientation_without_resizing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oriented.jpg"
    destination = tmp_path / "result.png"
    image = _pixels(17, 9)
    exif = image.getexif()
    exif[274] = 6
    image.save(source, quality=100, subsampling=0, exif=exif)
    with Image.open(source) as opened:
        expected = ImageOps.exif_transpose(opened).convert("RGB")
        expected_pixels = expected.tobytes()
        expected_size = expected.size

    result = process_image(source, destination)

    with Image.open(destination) as output:
        assert output.size == expected_size == (9, 17)
        assert output.tobytes() == expected_pixels
    assert (result.output_width, result.output_height) == expected_size


def test_process_image_rejects_invalid_content_and_animation(tmp_path: Path) -> None:
    invalid = tmp_path / "photo.jpg"
    invalid.write_text("this is not an image", encoding="utf-8")
    with pytest.raises(ImageInputError):
        process_image(invalid, tmp_path / "invalid.png")

    animated = tmp_path / "animated.webp"
    frames = [_pixels(3, 3), Image.new("RGB", (3, 3), (45, 15, 9))]
    frames[0].save(animated, format="WEBP", save_all=True, append_images=frames[1:])
    with pytest.raises(ImageInputError, match="animated"):
        process_image(animated, tmp_path / "animated.png")


def test_process_image_enforces_byte_pixel_and_output_limits(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _pixels(4, 3).save(source)

    with pytest.raises(ImageLimitError, match="byte limit"):
        process_image(
            source, tmp_path / "byte-limit.png", limits=ImageLimits(max_upload_bytes=1)
        )
    with pytest.raises(ImageLimitError, match="pixel limit"):
        process_image(
            source, tmp_path / "pixel-limit.png", limits=ImageLimits(max_pixels=11)
        )
    with pytest.raises(ImageInputError, match=".png"):
        process_image(source, tmp_path / "result.jpg")
    with pytest.raises(ImageInputError, match="different"):
        process_image(source, source)


def test_process_image_converts_unsupported_png_modes(tmp_path: Path) -> None:
    source = tmp_path / "cmyk.jpg"
    destination = tmp_path / "cmyk.png"
    Image.new("CMYK", (4, 3), (0, 128, 255, 0)).save(source)

    result = process_image(source, destination)

    with Image.open(destination) as output:
        assert output.mode == "RGB"
        assert output.size == (4, 3)
    assert result.output_mode == "RGB"
