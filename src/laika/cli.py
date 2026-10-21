"""Command-line interface for the shared Laika image engine."""

from __future__ import annotations

import argparse
import json
import sys

from laika.engine import ImageInputError, process_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laika")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process_parser = subparsers.add_parser(
        "process", help="decode, orient, and losslessly export a still image as PNG"
    )
    process_parser.add_argument("input", help="input JPEG, PNG, or WebP")
    process_parser.add_argument("output", help="output .png path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = process_image(args.input, args.output)
    except (ImageInputError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
