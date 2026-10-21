"""Incremental RGB8 PNG writing for outputs too large to keep in memory."""

from __future__ import annotations

import struct
import zlib
from typing import Any, BinaryIO

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_chunk(stream: BinaryIO, chunk_type: bytes, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)))
    stream.write(chunk_type)
    stream.write(payload)
    checksum = zlib.crc32(payload, zlib.crc32(chunk_type))
    stream.write(struct.pack(">I", checksum & 0xFFFFFFFF))


class PngRGBStreamWriter:
    """Write ordered RGB8 row blocks as a standards-compliant PNG stream."""

    def __init__(
        self,
        stream: BinaryIO,
        width: int,
        height: int,
        *,
        icc_profile: bytes | None = None,
        compression_level: int = 6,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("PNG dimensions must be positive")
        if not 0 <= compression_level <= 9:
            raise ValueError("PNG compression level must be between 0 and 9")
        self._stream = stream
        self._width = width
        self._height = height
        self._rows_written = 0
        self._compressor = zlib.compressobj(compression_level)
        self._finished = False

        stream.write(_PNG_SIGNATURE)
        _write_chunk(
            stream,
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        if icc_profile:
            payload = b"ICC Profile\0\0" + zlib.compress(icc_profile)
            _write_chunk(stream, b"iCCP", payload)

    def write_rows(self, rows: Any) -> None:
        """Append an RGB8 HWC block in top-to-bottom order."""
        if self._finished:
            raise ValueError("PNG stream has already been finalized")
        import numpy as np

        array = np.asarray(rows)
        if (
            array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[1:] != (self._width, 3)
            or array.shape[0] < 1
            or self._rows_written + array.shape[0] > self._height
        ):
            raise ValueError(
                "rows must be a non-empty uint8 RGB block matching PNG width"
            )

        row_bytes = self._width * 3
        filtered_stride = row_bytes + 1
        filtered = bytearray(filtered_stride * array.shape[0])
        filtered_view = memoryview(filtered)
        pixel_view = memoryview(np.ascontiguousarray(array)).cast("B")
        for row_index in range(array.shape[0]):
            filtered_offset = row_index * filtered_stride
            pixel_offset = row_index * row_bytes
            filtered_view[filtered_offset] = 0
            filtered_view[filtered_offset + 1 : filtered_offset + filtered_stride] = (
                pixel_view[pixel_offset : pixel_offset + row_bytes]
            )
        compressed = self._compressor.compress(filtered)
        if compressed:
            _write_chunk(self._stream, b"IDAT", compressed)
        self._rows_written += array.shape[0]

    def finish(self) -> None:
        """Write remaining compressed data and the PNG end marker."""
        if self._finished:
            raise ValueError("PNG stream has already been finalized")
        if self._rows_written != self._height:
            raise ValueError(
                f"PNG stream has {self._rows_written} of {self._height} rows"
            )
        compressed = self._compressor.flush()
        if compressed:
            _write_chunk(self._stream, b"IDAT", compressed)
        _write_chunk(self._stream, b"IEND", b"")
        self._finished = True


__all__ = ["PngRGBStreamWriter"]
