#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path


def read_config_ints(text: str):
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
            if isinstance(value, float):
                if not value.is_integer():
                    continue
                value = int(value)
            if isinstance(value, int):
                values[name] = value
                del pending[name]
                changed = True

    return values


def read_config_dims():
    cfg = Path(__file__).resolve().parents[1] / "aie" / "Config.h"
    text = cfg.read_text(encoding="utf-8")
    values = read_config_ints(text)
    required = (
        "kRows",
        "kCols",
        "kSimRows",
        "kRowDataElems",
        "kRowPhysElems",
        "kQ0PadIndex",
        "kFlushRows",
        "kRowsPerIterSim",
        "kSradIterations",
        "kSimIterations",
        "kSimInvalidRows",
        "kParallelLanes",
        "kBoardRows",
        "kBoardCols",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"cannot read {', '.join(missing)} from Config.h")
    lam_match = re.search(
        r"constexpr\s+float\s+kLambdaDefault\s*=\s*([0-9eE+\-.]+)f?\s*;",
        text,
    )
    lambda_default = float(lam_match.group(1)) if lam_match else 0.5
    bypass_match = re.search(
        r"constexpr\s+bool\s+kBypassCoeffMath\s*=\s*(true|false)\s*;",
        text,
    )
    bypass = bypass_match is not None and bypass_match.group(1) == "true"
    return (
        values["kRows"],
        values["kCols"],
        values["kSimRows"],
        values["kRowDataElems"],
        values["kRowPhysElems"],
        values["kQ0PadIndex"],
        values["kFlushRows"],
        values["kRowsPerIterSim"],
        values["kSradIterations"],
        values["kSimIterations"],
        values["kSimInvalidRows"],
        values["kParallelLanes"],
        values["kBoardRows"],
        values["kBoardCols"],
        lambda_default,
        bypass,
    )


(
    ROWS,
    COLS,
    SIM_ROWS,
    ROW_DATA,
    ROW_PHYS,
    Q0_PAD_INDEX,
    FLUSH_ROWS,
    ROWS_PER_ITER_SIM,
    SRAD_ITERATIONS,
    SIM_ITERATIONS,
    SIM_INVALID_ROWS,
    PARALLEL_LANES,
    BOARD_ROWS,
    BOARD_COLS,
    LAMBDA_DEFAULT,
    BYPASS_COEFF_MATH,
) = read_config_dims()
PIXELS = ROWS * COLS
BOARD_PIXELS = BOARD_ROWS * BOARD_COLS


def idx(r: int, c: int, cols: int = COLS) -> int:
    return r * cols + c


def compute_q0sqr(image, rows=ROWS):
    pixels = rows * COLS
    mean = sum(image[:pixels]) / pixels
    variance = sum(v * v for v in image[:pixels]) / pixels - mean * mean
    return variance / (mean * mean) if mean != 0.0 else 0.0


def clamp01(v):
    return max(0.0, min(1.0, v))


def sample_zero(image, r, c, rows=ROWS):
    if r < 0 or r >= rows or c < 0 or c >= COLS:
        return 0.0
    return image[idx(r, c)]


def compute_c(jc, d_n, d_s, d_w, d_e, q0sqr):
    if BYPASS_COEFF_MATH:
        return 1.0
    if q0sqr <= 0.0:
        return 1.0
    b = d_n + d_s + d_w + d_e
    dq = (jc + 0.25 * b) * (jc + 0.25 * b)
    nq = 0.5 * (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e)
    nq -= (1.0 / 16.0) * b * b
    num = q0sqr * (1.0 + q0sqr) * dq
    den = nq + q0sqr * q0sqr * dq
    if den == 0.0:
        return 0.0
    return clamp01(num / den)


def compute_c_zero_oob(image, r, c, q0sqr, rows=ROWS):
    jc = sample_zero(image, r, c, rows)
    d_n = sample_zero(image, r - 1, c, rows) - jc
    d_s = sample_zero(image, r + 1, c, rows) - jc
    d_w = sample_zero(image, r, c - 1, rows) - jc
    d_e = sample_zero(image, r, c + 1, rows) - jc
    return compute_c(jc, d_n, d_s, d_w, d_e, q0sqr)


def srad_next(image, q0sqr, rows=ROWS):
    out = [0.0] * (rows * COLS)
    for r in range(rows):
        for c in range(COLS):
            jc = sample_zero(image, r, c, rows)
            d_n = sample_zero(image, r - 1, c, rows) - jc
            d_s = sample_zero(image, r + 1, c, rows) - jc
            d_w = sample_zero(image, r, c - 1, rows) - jc
            d_e = sample_zero(image, r, c + 1, rows) - jc
            coeff = compute_c_zero_oob(image, r, c, q0sqr, rows)
            coeff_south = compute_c_zero_oob(image, r + 1, c, q0sqr, rows)
            coeff_east = compute_c_zero_oob(image, r, c + 1, q0sqr, rows)
            divergence = (
                coeff * d_n
                + coeff_south * d_s
                + coeff * d_w
                + coeff_east * d_e
            )
            out[idx(r, c)] = jc + 0.25 * LAMBDA_DEFAULT * divergence
    return out


def write_plio64_stream(path: Path, data):
    if len(data) % 2 != 0:
        raise ValueError(f"{path} requires an even float count for plio_64_bits")
    with path.open("w", encoding="utf-8") as f:
        for i in range(0, len(data), 2):
            f.write(f"{data[i]:.9g} {data[i + 1]:.9g}\n")


def write_matrix(path: Path, data, rows: int, cols: int):
    if len(data) != rows * cols:
        raise ValueError(f"{path} got {len(data)} floats, expected {rows * cols}")
    with path.open("w", encoding="utf-8") as f:
        for r in range(rows):
            row = data[r * cols : (r + 1) * cols]
            f.write(" ".join(f"{v:.9g}" for v in row) + "\n")


def default_pixel(r: int, c: int) -> float:
    return (
        1.0
        + 0.003 * r
        + 0.002 * c
        + 0.05 * math.sin(0.31 * r) * math.cos(0.19 * c)
    )


def write_default_image_matrix(path: Path, rows: int, cols: int):
    with path.open("w", encoding="utf-8") as f:
        for r in range(rows):
            row = (f"{default_pixel(r, c):.9g}" for c in range(cols))
            f.write(" ".join(row) + "\n")


def make_rowstream_input(image, q0sqr, n_stream_rows, valid_rows=ROWS):
    out = []
    for r in range(n_stream_rows):
        row = [0.0] * ROW_PHYS
        if r < valid_rows:
            row[:ROW_DATA] = image[r * COLS : r * COLS + ROW_DATA]
            row[Q0_PAD_INDEX] = q0sqr
        out.extend(row)
    return out


def make_stats_rowstream_input(image, n_stream_rows, valid_rows=ROWS):
    out = []
    for r in range(n_stream_rows):
        row = [0.0] * ROW_PHYS
        if r < valid_rows:
            row[:ROW_DATA] = image[r * COLS : r * COLS + ROW_DATA]
        out.extend(row)
    return out


STALE_FILES = (
    "aie_j_next.txt",
    "aiesimoutput.txt",
    "aiesim_j_next.txt",
    "aiesim_row_stats.txt",
    "gold_srad_no_boundary.txt",
    "gold_srad.txt",
    "gold_srad_tile.txt",
    "gold_srad_rowstream.txt",
    "golden_32x32.txt",
    "input_32x32.txt",
    "input_image_sim.txt",
    "lambda.txt",
    "plio_ours_j.txt",
    "plio_ours_j_stats.txt",
    "plio_ours_j_next.txt",
    "plio_ours_j_next_0.txt",
    "plio_ours_j_tile.txt",
    "plio_ours_j_tile.txt.tmp",
    "plio_ours_j_tile_0.txt",
    "plio_ours_j_tile_0.txt.tmp",
    "plio_ours_j_tile_1.txt",
    "plio_ours_j_tile_1.txt.tmp",
    "plio_ours_j_tile_2.txt",
    "plio_ours_j_tile_2.txt.tmp",
    "plio_ours_j_tile_3.txt",
    "plio_ours_j_tile_3.txt.tmp",
    "plio_ours_j.txt.tmp",
    "q0sqr.txt",
    "q0sqr_ref.txt",
)


def cleanup_stale_files(base: Path):
    for name in STALE_FILES:
        path = base / name
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass


def make_default_image(rows: int = ROWS, cols: int = COLS):
    image = []
    for r in range(rows):
        for c in range(cols):
            image.append(default_pixel(r, c))
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate Ours SRAD PLIO data.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="also generate small AIE simulation input and PLIO row stream",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    cleanup_stale_files(base)

    write_default_image_matrix(base / "input_image.txt", BOARD_ROWS, BOARD_COLS)

    if not args.sim:
        if args.verbose:
            print(
                f"generated board input {BOARD_ROWS}x{BOARD_COLS} "
                f"({BOARD_PIXELS} floats)"
            )
        return

    image = make_default_image()
    write_matrix(base / "input_image_sim.txt", image, ROWS, COLS)

    stats_stream = []
    update_stream = []
    q0_values = []
    cur = image[: SIM_ROWS * COLS]
    for _ in range(SIM_ITERATIONS):
        stats_stream.extend(
            make_stats_rowstream_input(cur, ROWS_PER_ITER_SIM, SIM_ROWS)
        )
        q0sqr = compute_q0sqr(cur, SIM_ROWS)
        q0_values.append(q0sqr)
        update_stream.extend(
            make_rowstream_input(cur, q0sqr, ROWS_PER_ITER_SIM, SIM_ROWS)
        )
        cur = srad_next(cur, q0sqr, SIM_ROWS)

    write_plio64_stream(base / "plio_ours_j_stats.txt", stats_stream)
    write_plio64_stream(base / "plio_ours_j.txt", update_stream)
    if not args.verbose:
        return

    print(f"generated board input {BOARD_ROWS}x{BOARD_COLS} ({BOARD_PIXELS} floats)")
    print(f"generated sim input {ROWS}x{COLS} Ours row-stream SRAD case")
    print(
        f"sim rows/iter={SIM_ROWS}, physical row={ROW_PHYS}, "
        f"invalid output rows/iter={SIM_INVALID_ROWS}, "
        f"sim iterations={SIM_ITERATIONS}"
    )
    print(
        f"lambda={LAMBDA_DEFAULT:.9g}, "
        f"bypass_coeff_math={BYPASS_COEFF_MATH}, lanes={PARALLEL_LANES}"
    )
    print(
        f"PLIO lines: stats_input={len(stats_stream) // 2}, "
        f"update_input={len(update_stream) // 2}"
    )
    for i, q0 in enumerate(q0_values):
        print(f"q0sqr iter{i}={q0:.9g}")
    print(f"q0sqr first={q0_values[0]:.9g}, last={q0_values[-1]:.9g}")


if __name__ == "__main__":
    main()
