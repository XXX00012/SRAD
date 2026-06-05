#!/usr/bin/env python3

import math
import re
from pathlib import Path


def read_config():
    cfg = Path(__file__).resolve().parents[1] / "aie" / "Config.h"
    text = cfg.read_text(encoding="utf-8")
    rows = re.search(r"constexpr\s+int\s+kRows\s*=\s*(\d+)\s*;", text)
    cols = re.search(r"constexpr\s+int\s+kCols\s*=\s*(\d+)\s*;", text)
    block_rows = re.search(r"constexpr\s+int\s+kBlockRows\s*=\s*(\d+)\s*;", text)
    block_cols = re.search(r"constexpr\s+int\s+kBlockCols\s*=\s*(\d+)\s*;", text)
    lanes = re.search(r"constexpr\s+int\s+kLanes\s*=\s*(\d+)\s*;", text)
    if (
        rows is None
        or cols is None
        or block_rows is None
        or block_cols is None
        or lanes is None
    ):
        raise RuntimeError(f"cannot read SRAD dimensions from {cfg}")
    return (
        int(rows.group(1)),
        int(cols.group(1)),
        int(block_rows.group(1)),
        int(block_cols.group(1)),
        int(lanes.group(1)),
    )


ROWS, COLS, BLOCK_ROWS, BLOCK_COLS, LANES = read_config()
PIXELS = ROWS * COLS
BLOCK_COUNT = (ROWS // BLOCK_ROWS) * (COLS // BLOCK_COLS)
LAMBDA = 0.5


def block_north_row(r: int) -> int:
    return 0 if r == 0 else r - 1


def block_south_row(r: int) -> int:
    return BLOCK_ROWS - 1 if r == BLOCK_ROWS - 1 else r + 1


def block_west_col(c: int) -> int:
    return 0 if c == 0 else c - 1


def block_east_col(c: int) -> int:
    return BLOCK_COLS - 1 if c == BLOCK_COLS - 1 else c + 1


def idx(r: int, c: int) -> int:
    return r * COLS + c


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_q0sqr(image):
    mean = sum(image) / PIXELS
    variance = sum(v * v for v in image) / PIXELS - mean * mean
    return variance / (mean * mean) if mean != 0.0 else 0.0


def compute_c(jc, d_n, d_s, d_w, d_e, q0sqr):
    g2 = (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e) / (jc * jc)
    lap = (d_n + d_s + d_w + d_e) / jc
    num = 0.5 * g2 - (1.0 / 16.0) * lap * lap
    den = 1.0 + 0.25 * lap
    qsqr = num / (den * den)
    den = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr))
    c = 1.0 / (1.0 + den)
    return clamp01(c)


def update_j(jc, d_val, lam):
    return jc + 0.25 * lam * d_val


def srad_one_iteration(image, q0sqr, lam):
    c_plane = [0.0] * PIXELS

    for br in range(0, ROWS, BLOCK_ROWS):
        for bc in range(0, COLS, BLOCK_COLS):
            for tr in range(BLOCK_ROWS):
                for tc in range(BLOCK_COLS):
                    r = br + tr
                    c = bc + tc
                    p = idx(r, c)
                    jc = image[p]
                    d_n = image[idx(br + block_north_row(tr), c)] - jc
                    d_s = image[idx(br + block_south_row(tr), c)] - jc
                    d_w = image[idx(r, bc + block_west_col(tc))] - jc
                    d_e = image[idx(r, bc + block_east_col(tc))] - jc
                    c_plane[p] = compute_c(jc, d_n, d_s, d_w, d_e, q0sqr)

    out = [0.0] * PIXELS
    for br in range(0, ROWS, BLOCK_ROWS):
        for bc in range(0, COLS, BLOCK_COLS):
            for tr in range(BLOCK_ROWS):
                for tc in range(BLOCK_COLS):
                    r = br + tr
                    c = bc + tc
                    p = idx(r, c)
                    jc = image[p]
                    d_n = image[idx(br + block_north_row(tr), c)] - jc
                    d_s = image[idx(br + block_south_row(tr), c)] - jc
                    d_w = image[idx(r, bc + block_west_col(tc))] - jc
                    d_e = image[idx(r, bc + block_east_col(tc))] - jc
                    d_val = (
                        c_plane[p] * d_n
                        + c_plane[idx(br + block_south_row(tr), c)] * d_s
                        + c_plane[p] * d_w
                        + c_plane[idx(r, bc + block_east_col(tc))] * d_e
                    )
                    out[p] = update_j(image[p], d_val, lam)

    return out


def write_matrix(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for r in range(ROWS):
            row = data[r * COLS : (r + 1) * COLS]
            f.write(" ".join(f"{v:.9g}" for v in row))
            f.write("\n")


def pack_blocks(image):
    packed = []
    for br in range(0, ROWS, BLOCK_ROWS):
        for bc in range(0, COLS, BLOCK_COLS):
            for r in range(BLOCK_ROWS):
                for c in range(BLOCK_COLS):
                    packed.append(image[idx(br + r, bc + c)])
    return packed


def write_stream(path: Path, data):
    path.write_text("".join(f"{v:.9g}\n" for v in data), encoding="utf-8")


def scalar_packets(value):
    data = [0.0] * (BLOCK_COUNT * LANES)
    for blk in range(BLOCK_COUNT):
        data[blk * LANES] = value
    return data


def main():
    base = Path(__file__).resolve().parent
    image = []
    for r in range(ROWS):
        for c in range(COLS):
            v = (
                1.0
                + 0.003 * r
                + 0.002 * c
                + 0.05 * math.sin(0.31 * r) * math.cos(0.19 * c)
            )
            image.append(v)

    q0sqr = compute_q0sqr(image)
    golden = srad_one_iteration(image, q0sqr, LAMBDA)

    write_matrix(base / "input_32x32.txt", image)
    write_matrix(base / "golden_32x32.txt", golden)
    q0_packet = "\n".join(f"{q0sqr:.9g}" if i == 0 else "0" for i in range(LANES)) + "\n"
    (base / "q0sqr_ref.txt").write_text(q0_packet, encoding="utf-8")
    (base / "q0sqr.txt").write_text(q0_packet, encoding="utf-8")
    (base / "lambda.txt").write_text(f"{LAMBDA:.9g}\n", encoding="utf-8")
    packed = pack_blocks(image)
    write_stream(base / "plio_ours_undivide_j.txt", packed)
    write_stream(base / "plio_ours_undivide_j_update.txt", packed)
    write_stream(base / "plio_ours_undivide_q0sqr.txt", scalar_packets(q0sqr))
    write_stream(base / "plio_ours_undivide_lambda.txt", scalar_packets(LAMBDA))
    print(f"generated {ROWS}x{COLS} Ours float SRAD case")
    print(f"q0sqr={q0sqr:.9g} lambda={LAMBDA:.9g}")


if __name__ == "__main__":
    main()
