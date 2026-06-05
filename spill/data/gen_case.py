#!/usr/bin/env python3

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
        "kGraphRunIterations",
        "kOutputTileRows",
        "kOutputTileCols",
        "kHaloTopRows",
        "kHaloLeftCols",
        "kInputLogicalRows",
        "kInputLogicalCols",
        "kInputRowElems",
        "kQ0SqrTileRow",
        "kQ0SqrTileCol",
        "kTileRowCount",
        "kTileColCount",
        "kTileStrideRows",
        "kTileStrideCols",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"cannot read {', '.join(missing)} from Config.h")
    return (
        values["kRows"],
        values["kCols"],
        values["kGraphRunIterations"],
        values["kOutputTileRows"],
        values["kOutputTileCols"],
        values["kHaloTopRows"],
        values["kHaloLeftCols"],
        values["kInputLogicalRows"],
        values["kInputLogicalCols"],
        values["kInputRowElems"],
        values["kQ0SqrTileRow"],
        values["kQ0SqrTileCol"],
        values["kTileRowCount"],
        values["kTileColCount"],
        values["kTileStrideRows"],
        values["kTileStrideCols"],
    )


(
    ROWS,
    COLS,
    GRAPH_RUN_ITERATIONS,
    OUTPUT_TILE_ROWS,
    OUTPUT_TILE_COLS,
    HALO_TOP_ROWS,
    HALO_LEFT_COLS,
    INPUT_LOGICAL_ROWS,
    INPUT_LOGICAL_COLS,
    INPUT_ROW_ELEMS,
    Q0SQR_TILE_ROW,
    Q0SQR_TILE_COL,
    TILE_ROW_COUNT,
    TILE_COL_COUNT,
    TILE_STRIDE_ROWS,
    TILE_STRIDE_COLS,
) = read_config_dims()
PIXELS = ROWS * COLS


def idx(r: int, c: int) -> int:
    return r * COLS + c


def compute_q0sqr(image):
    mean = sum(image) / PIXELS
    variance = sum(v * v for v in image) / PIXELS - mean * mean
    return variance / (mean * mean) if mean != 0.0 else 0.0


def write_plio64_stream(path: Path, data):
    if len(data) % 2 != 0:
        raise ValueError(f"{path} requires an even float count for plio_64_bits")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i in range(0, len(data), 2):
            f.write(f"{data[i]:.9g} {data[i + 1]:.9g}\n")
    tmp.replace(path)


def make_j_stream(image):
    out = []
    for r in range(ROWS):
        start = idx(r, 0)
        out.extend(image[start : start + COLS])
    return out


def tile_start(tile_idx: int, stride: int) -> int:
    return tile_idx * stride


def make_j_tile_stream(image, q0sqr):
    out = []
    for tile_r in range(TILE_ROW_COUNT):
        row_start = tile_start(tile_r, TILE_STRIDE_ROWS)
        for tile_c in range(TILE_COL_COUNT):
            col_start = tile_start(tile_c, TILE_STRIDE_COLS)
            for lr in range(INPUT_LOGICAL_ROWS):
                gr = row_start + lr - HALO_TOP_ROWS
                for lc in range(INPUT_ROW_ELEMS):
                    if lr == Q0SQR_TILE_ROW and lc == Q0SQR_TILE_COL:
                        out.append(q0sqr)
                        continue
                    if lc >= INPUT_LOGICAL_COLS:
                        out.append(0.0)
                        continue
                    gc = col_start + lc - HALO_LEFT_COLS
                    if 0 <= gr < ROWS and 0 <= gc < COLS:
                        out.append(image[idx(gr, gc)])
                    else:
                        out.append(0.0)
    return out


STALE_FILES = (
    "aie_j_next.txt",
    "aiesimoutput.txt",
    "aiesim_j_next.txt",
    "gold_srad_no_boundary.txt",
    "gold_srad.txt",
    "gold_srad_tile.txt",
    "golden_32x32.txt",
    "input_32x32.txt",
    "lambda.txt",
    "plio_ours_j_next.txt",
    "plio_ours_j_tile.txt",
    "plio_ours_j_tile.txt.tmp",
    "plio_ours_j_k1.txt",
    "plio_ours_j_k1.txt.tmp",
    "plio_ours_j_k2.txt",
    "plio_ours_j_k2.txt.tmp",
    "plio_ours_j_update.txt",
    "plio_ours_lambda.txt",
    "plio_ours_j.txt.tmp",
    "plio_ours_q0sqr.txt",
    "plio_ours_q0sqr.txt.tmp",
    "q0sqr.txt",
    "q0sqr_ref.txt",
)


def cleanup_stale_files(base: Path):
    for name in STALE_FILES:
        path = base / name
        if path.exists():
            path.unlink()


def main():
    base = Path(__file__).resolve().parent
    cleanup_stale_files(base)

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

    j_ddr_stream = make_j_stream(image)
    j_tile_stream = make_j_tile_stream(image, q0sqr)
    write_plio64_stream(base / "plio_ours_j.txt", j_ddr_stream)
    write_plio64_stream(base / "plio_ours_j_tile.txt", j_tile_stream)
    print(f"generated {ROWS}x{COLS} Ours float SRAD case")
    print(
        f"tile stream={TILE_ROW_COUNT}x{TILE_COL_COUNT} tiles, "
        f"output {OUTPUT_TILE_ROWS}x{OUTPUT_TILE_COLS}, "
        f"input logical {INPUT_LOGICAL_ROWS}x{INPUT_LOGICAL_COLS}, "
        f"physical row {INPUT_ROW_ELEMS}"
    )
    print(
        f"graph firings={GRAPH_RUN_ITERATIONS}, "
        f"input floats/firing={INPUT_LOGICAL_ROWS * INPUT_ROW_ELEMS}, "
        f"output floats/firing={OUTPUT_TILE_ROWS * OUTPUT_TILE_COLS}"
    )
    print(
        f"PLIO lines: J_tile={len(j_tile_stream) // 2}; "
        f"q0 embedded at row {Q0SQR_TILE_ROW}, col {Q0SQR_TILE_COL}"
    )
    print(f"q0sqr={q0sqr:.9g}")


if __name__ == "__main__":
    main()
