#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path

LAMBDA = 0.5


def read_config_ints():
    cfg = Path(__file__).resolve().parents[1] / "aie" / "Config.h"
    text = cfg.read_text(encoding="utf-8")
    values = {}
    pending = {}
    for name, expr in re.findall(r"constexpr\s+int\s+(\w+)\s*=\s*([^;]+)\s*;", text):
        expr = expr.strip()
        if re.fullmatch(r"\d+", expr):
            values[name] = int(expr)
        else:
            pending[name] = expr

    changed = True
    while changed:
        changed = False
        for name, expr in list(pending.items()):
            expr_py = expr.replace("sizeof(float)", "4").replace("/", "//")
            if not re.fullmatch(r"[\w\s+\-*/()%]+", expr_py):
                continue
            try:
                value = eval(expr_py, {"__builtins__": {}}, values)
            except NameError:
                continue
            if isinstance(value, int):
                values[name] = value
                del pending[name]
                changed = True

    required = (
        "kRows",
        "kCols",
        "kPixels",
        "kTileRows",
        "kTileCols",
        "kTileLeftHaloCols",
        "kInputTileRows",
        "kInputTileCols",
        "kColTiles",
        "kTilesPerIteration",
        "kParamElems",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"cannot read {', '.join(missing)} from Config.h")
    return values


CFG = read_config_ints()
ROWS = CFG["kRows"]
COLS = CFG["kCols"]
PIXELS = CFG["kPixels"]
DIM_TAG = f"{ROWS}x{COLS}"
INPUT_NAME = f"input_{DIM_TAG}.txt"
GOLDEN_NAME = f"golden_{DIM_TAG}.txt"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def idx(row, col):
    return row + ROWS * col


def north_row(row):
    return 0 if row == 0 else row - 1


def south_row(row):
    return ROWS - 1 if row == ROWS - 1 else row + 1


def west_col(col):
    return 0 if col == 0 else col - 1


def east_col(col):
    return COLS - 1 if col == COLS - 1 else col + 1


def make_image():
    image = [0.0] * PIXELS
    for row in range(ROWS):
        for col in range(COLS):
            image[idx(row, col)] = (
                1.0
                + 0.003 * row
                + 0.002 * col
                + 0.05 * math.sin(0.31 * row) * math.cos(0.19 * col)
            )
    return image


def compute_q0sqr(image):
    mean = sum(image) / PIXELS
    variance = sum(v * v for v in image) / PIXELS - mean * mean
    return variance / (mean * mean) if mean != 0.0 else 0.0


def clamp01(value):
    return max(0.0, min(1.0, value))


def compute_c(jc, d_n, d_s, d_w, d_e, q0sqr):
    g2 = (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e) / (jc * jc)
    lap = (d_n + d_s + d_w + d_e) / jc
    num = 0.5 * g2 - (1.0 / 16.0) * lap * lap
    den = 1.0 + 0.25 * lap
    qsqr = num / (den * den)
    den = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr))
    return clamp01(1.0 / (1.0 + den))


def srad_one_iteration(image, q0sqr, lam):
    coeff = [0.0] * PIXELS

    for row in range(ROWS):
        for col in range(COLS):
            p = idx(row, col)
            jc = image[p]
            d_n = image[idx(north_row(row), col)] - jc
            d_s = image[idx(south_row(row), col)] - jc
            d_w = image[idx(row, west_col(col))] - jc
            d_e = image[idx(row, east_col(col))] - jc
            coeff[p] = compute_c(jc, d_n, d_s, d_w, d_e, q0sqr)

    out = [0.0] * PIXELS
    for row in range(ROWS):
        for col in range(COLS):
            p = idx(row, col)
            jc = image[p]
            d_n = image[idx(north_row(row), col)] - jc
            d_s = image[idx(south_row(row), col)] - jc
            d_w = image[idx(row, west_col(col))] - jc
            d_e = image[idx(row, east_col(col))] - jc
            d_val = (
                coeff[p] * d_n
                + coeff[idx(south_row(row), col)] * d_s
                + coeff[p] * d_w
                + coeff[idx(row, east_col(col))] * d_e
            )
            out[p] = jc + 0.25 * lam * d_val

    return out


def tile_start_row(tile):
    return (tile // CFG["kColTiles"]) * CFG["kTileRows"]


def tile_start_col(tile):
    return (tile % CFG["kColTiles"]) * CFG["kTileCols"]


def make_compute_stream(image):
    stream = []
    stream.extend(image)

    for tile in range(CFG["kTilesPerIteration"]):
        start_row = tile_start_row(tile)
        start_col = tile_start_col(tile)
        for local_row in range(CFG["kInputTileRows"]):
            row = clamp(start_row + local_row - 1, 0, ROWS - 1)
            for local_col in range(CFG["kInputTileCols"]):
                col = clamp(
                    start_col + local_col - CFG["kTileLeftHaloCols"],
                    0,
                    COLS - 1,
                )
                stream.append(image[idx(row, col)])

    return stream


def write_matrix(path, data):
    with path.open("w", encoding="utf-8") as fout:
        for row in range(ROWS):
            values = [data[idx(row, col)] for col in range(COLS)]
            fout.write(" ".join(f"{value:.9g}" for value in values))
            fout.write("\n")


def write_vector(path, data):
    path.write_text(
        "\n".join(f"{value:.9g}" for value in data) + "\n",
        encoding="utf-8",
    )


def main():
    sim_runtime = "--sim-runtime" in sys.argv[1:]
    base = Path(__file__).resolve().parent
    image = make_image()

    write_matrix(base / INPUT_NAME, image)
    write_vector(base / "lambda.txt", [LAMBDA])

    if sim_runtime:
        q0sqr = compute_q0sqr(image)
        golden = srad_one_iteration(image, q0sqr, LAMBDA)
        params = [LAMBDA, 0.0]
        while len(params) < CFG["kParamElems"]:
            params.append(0.0)

        write_matrix(base / GOLDEN_NAME, golden)
        write_vector(base / "plio_fpga_params.txt", params)
        write_vector(base / "plio_fpga_compute.txt", make_compute_stream(image))
        write_vector(base / "q0sqr_ref.txt", [q0sqr])
        print(f"generated {ROWS}x{COLS} srad_fpga board/AIE sim case")
        print(
            f"tiles={CFG['kTilesPerIteration']} q0sqr={q0sqr:.9g} "
            f"lambda={LAMBDA:.9g}"
        )
    else:
        print(f"generated {ROWS}x{COLS} srad_fpga board input")
        print(
            "sim PLIO/golden files skipped; rerun with --sim-runtime "
            "when a full AIE simulation data set is needed"
        )


if __name__ == "__main__":
    main()
