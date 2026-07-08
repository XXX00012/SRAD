#!/usr/bin/env python3
"""Generate the SRAD v1 input image in ASCII PGM format."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SRAD input image.pgm.")
    parser.add_argument("--rows", type=int, default=502, help="image rows")
    parser.add_argument("--cols", type=int, default=458, help="image columns")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("image.pgm"),
        help="output PGM path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("rows and cols must be positive")

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="ascii") as f:
        f.write("P2\n")
        f.write(f"{args.cols} {args.rows}\n")
        f.write("255\n")
        for _ in range(args.rows):
            row = [str(random.randint(0, 255)) for _ in range(args.cols)]
            f.write(" ".join(row) + "\n")

    print(f"wrote {args.output}: {args.rows}x{args.cols}, seed={args.seed}")


if __name__ == "__main__":
    main()
