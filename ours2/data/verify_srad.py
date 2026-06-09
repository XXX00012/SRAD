#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_config_ints(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    pending: dict[str, str] = {}
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


def read_config() -> tuple[dict[str, int], float, bool]:
    cfg_path = Path(__file__).resolve().parents[1] / "aie" / "Config.h"
    text = cfg_path.read_text(encoding="utf-8")
    values = read_config_ints(text)
    required = (
        "kRows",
        "kCols",
        "kOutputTileRows",
        "kOutputTileCols",
        "kHaloTopRows",
        "kOutputSampleElems",
        "kOutputTileDataElems",
        "kOutputStatElems",
        "kTileRowCount",
        "kTileColCount",
        "kTileStrideRows",
        "kTileStrideCols",
        "kGraphRunIterations",
        "kDefaultIterations",
        "kOutputFirstRow",
        "kOutputLastRow",
        "kOutputFirstCol",
        "kOutputLastCol",
        "kOutputRows",
        "kOutputCols",
        "kCompareRows",
        "kParallelLanes",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"cannot read {', '.join(missing)} from {cfg_path}")

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

    return values, lambda_default, bypass


def read_floats(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [float(tok) for tok in path.read_text(encoding="utf-8").split()]


def read_scalar(path: Path, default: float) -> float:
    if not path.exists():
        return default
    values = read_floats(path)
    return values[0] if values else default


def write_plio64_stream(path: Path, values: list[float]) -> None:
    if len(values) % 2 != 0:
        raise ValueError(f"{path} requires an even float count for plio_64_bits")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for i in range(0, len(values), 2):
            fout.write(f"{values[i]:.9g} {values[i + 1]:.9g}\n")


def write_matrix(path: Path, values: list[float], rows: int, cols: int) -> None:
    if len(values) != rows * cols:
        raise ValueError(f"{path} got {len(values)} floats, expected {rows * cols}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for r in range(rows):
            row = values[r * cols:(r + 1) * cols]
            fout.write(" ".join(f"{v:.9g}" for v in row) + "\n")


def compact_to_raw_tile_stream(compact: list[float],
                               cfg: dict[str, int],
                               include_stats: bool = True) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    if len(compact) != output_rows * output_cols:
        raise ValueError(
            f"compact gold has {len(compact)} floats, "
            f"expected {output_rows * output_cols}"
        )

    raw: list[float] = []
    for tile_r in range(cfg["kTileRowCount"]):
        tile_row_start = tile_start(tile_r, cfg["kTileStrideRows"])
        for tile_c in range(cfg["kTileColCount"]):
            tile_col_start = tile_start(tile_c, cfg["kTileStrideCols"])
            tile_sum = 0.0
            tile_sum2 = 0.0
            for local_row in range(cfg["kOutputTileRows"]):
                global_row = tile_row_start + local_row
                for local_col in range(cfg["kOutputTileCols"]):
                    global_col = tile_col_start + local_col
                    if not (cfg["kOutputFirstRow"] <= global_row <= cfg["kOutputLastRow"]):
                        value = 0.0
                        raw.append(value)
                        continue
                    if not (cfg["kOutputFirstCol"] <= global_col <= cfg["kOutputLastCol"]):
                        value = 0.0
                        raw.append(value)
                        continue
                    out_r = global_row - cfg["kOutputFirstRow"]
                    out_c = global_col - cfg["kOutputFirstCol"]
                    value = compact[out_r * output_cols + out_c]
                    raw.append(value)
                    tile_sum += value
                    tile_sum2 += value * value
            if include_stats:
                raw.append(tile_sum)
                raw.append(tile_sum2)
    return raw


def read_input_image(path: Path, base: Path, rows: int, cols: int) -> list[float]:
    pixels = rows * cols
    if path.exists():
        image = read_floats(path)
        if len(image) != pixels:
            raise RuntimeError(f"{path} has {len(image)} floats, expected {pixels}")
        return image

    plio_path = base / "plio_ours_j.txt"
    image = read_floats(plio_path)
    if len(image) != pixels:
        raise RuntimeError(f"{plio_path} has {len(image)} floats, expected {pixels}")
    return image


def idx(r: int, c: int, cols: int) -> int:
    return r * cols + c


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def compute_q0sqr(image: list[float]) -> float:
    mean = sum(image) / len(image)
    variance = sum(v * v for v in image) / len(image) - mean * mean
    return variance / (mean * mean) if mean != 0.0 else 0.0


def compute_c(jc: float,
              d_n: float,
              d_s: float,
              d_w: float,
              d_e: float,
              q0sqr: float,
              bypass: bool) -> float:
    if bypass:
        return 1.0
    q0_den = q0sqr * (1.0 + q0sqr)
    if jc == 0.0 or q0_den == 0.0:
        return 1.0
    g2 = (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e) / (jc * jc)
    lap = (d_n + d_s + d_w + d_e) / jc
    num = 0.5 * g2 - (1.0 / 16.0) * lap * lap
    den = 1.0 + 0.25 * lap
    if den == 0.0:
        return 0.0
    qsqr = num / (den * den)
    coeff_den = 1.0 + (qsqr - q0sqr) / q0_den
    if coeff_den == 0.0:
        return 0.0
    return clamp01(1.0 / coeff_den)


def sample_zero(image: list[float], r: int, c: int, rows: int, cols: int) -> float:
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0.0
    return image[idx(r, c, cols)]


def compute_c_zero_oob(image: list[float],
                       r: int,
                       c: int,
                       rows: int,
                       cols: int,
                       q0sqr: float,
                       bypass: bool) -> float:
    jc = sample_zero(image, r, c, rows, cols)
    d_n = sample_zero(image, r - 1, c, rows, cols) - jc
    d_s = sample_zero(image, r + 1, c, rows, cols) - jc
    d_w = sample_zero(image, r, c - 1, rows, cols) - jc
    d_e = sample_zero(image, r, c + 1, rows, cols) - jc
    return compute_c(jc, d_n, d_s, d_w, d_e, q0sqr, bypass)


def srad_compact(image: list[float],
                 rows: int,
                 cols: int,
                 output_first_row: int,
                 output_last_row: int,
                 output_first_col: int,
                 output_last_col: int,
                 lam: float,
                 q0sqr: float,
                 bypass: bool) -> list[float]:
    output_rows = output_last_row - output_first_row + 1
    output_cols = output_last_col - output_first_col + 1
    out = [0.0] * (output_rows * output_cols)
    for r in range(output_first_row, output_last_row + 1):
        for c in range(output_first_col, output_last_col + 1):
            jc = sample_zero(image, r, c, rows, cols)
            d_n = sample_zero(image, r - 1, c, rows, cols) - jc
            d_s = sample_zero(image, r + 1, c, rows, cols) - jc
            d_w = sample_zero(image, r, c - 1, rows, cols) - jc
            d_e = sample_zero(image, r, c + 1, rows, cols) - jc
            coeff = compute_c_zero_oob(image, r, c, rows, cols, q0sqr, bypass)
            coeff_south = compute_c_zero_oob(
                image, r + 1, c, rows, cols, q0sqr, bypass
            )
            coeff_east = compute_c_zero_oob(
                image, r, c + 1, rows, cols, q0sqr, bypass
            )
            divergence = (
                coeff * d_n
                + coeff_south * d_s
                + coeff * d_w
                + coeff_east * d_e
            )
            out_r = r - output_first_row
            out_c = c - output_first_col
            out[out_r * output_cols + out_c] = jc + 0.25 * lam * divergence

    return out


def srad_multi_iteration(image: list[float],
                         cfg: dict[str, int],
                         lam: float,
                         bypass: bool,
                         iters: int) -> tuple[list[float], list[float], list[float]]:
    cur = image
    q0_values: list[float] = []
    raw_all_iters: list[float] = []

    for _ in range(iters):
        q0sqr = compute_q0sqr(cur)
        q0_values.append(q0sqr)
        cur = srad_compact(
            cur,
            cfg["kRows"],
            cfg["kCols"],
            cfg["kOutputFirstRow"],
            cfg["kOutputLastRow"],
            cfg["kOutputFirstCol"],
            cfg["kOutputLastCol"],
            lam,
            q0sqr,
            bypass,
        )
        raw_all_iters.extend(compact_to_raw_tile_stream(cur, cfg))

    return cur, q0_values, raw_all_iters


def tile_start(tile_idx: int, stride: int) -> int:
    return tile_idx * stride


def normalize_aie_output(values: list[float],
                         cfg: dict[str, int],
                         iters: int,
                         force_raw_tile_stream: bool = False) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    compact_elems = output_rows * output_cols
    if len(values) == compact_elems and not force_raw_tile_stream:
        return values

    raw_one_iter_elems = cfg["kGraphRunIterations"] * cfg["kOutputSampleElems"]
    expected_raw_elems = raw_one_iter_elems * iters
    if len(values) == expected_raw_elems or len(values) == raw_one_iter_elems:
        iter_offset = 0 if len(values) == raw_one_iter_elems else (iters - 1) * raw_one_iter_elems
        out = [0.0] * compact_elems
        pos = iter_offset
        for tile_r in range(cfg["kTileRowCount"]):
            tile_row_start = tile_start(tile_r, cfg["kTileStrideRows"])
            for tile_c in range(cfg["kTileColCount"]):
                tile_col_start = tile_start(tile_c, cfg["kTileStrideCols"])
                tile_base = pos
                data_pos = tile_base
                for local_row in range(cfg["kOutputTileRows"]):
                    for local_col in range(cfg["kOutputTileCols"]):
                        value = values[data_pos]
                        data_pos += 1
                        global_row = tile_row_start + local_row
                        global_col = tile_col_start + local_col
                        if not (cfg["kOutputFirstRow"] <= global_row <= cfg["kOutputLastRow"]):
                            continue
                        if not (cfg["kOutputFirstCol"] <= global_col <= cfg["kOutputLastCol"]):
                            continue
                        out_r = global_row - cfg["kOutputFirstRow"]
                        out_c = global_col - cfg["kOutputFirstCol"]
                        out[out_r * output_cols + out_c] = value
                pos = tile_base + cfg["kOutputSampleElems"]
        return out

    lane_major_one_iter_elems = (
        cfg["kParallelLanes"] *
        cfg["kGraphRunIterations"] *
        cfg["kOutputSampleElems"]
    )
    expected_lane_major_elems = lane_major_one_iter_elems * iters
    if len(values) == expected_lane_major_elems:
        out = [0.0] * compact_elems
        lane_span = cfg["kGraphRunIterations"] * cfg["kOutputSampleElems"] * iters
        iter_span = cfg["kGraphRunIterations"] * cfg["kOutputSampleElems"]
        for lane in range(cfg["kParallelLanes"]):
            lane_base = lane * lane_span + (iters - 1) * iter_span
            for graph_iter in range(cfg["kGraphRunIterations"]):
                tile_linear = graph_iter * cfg["kParallelLanes"] + lane
                if tile_linear >= cfg["kTileRowCount"] * cfg["kTileColCount"]:
                    continue
                tile_r = tile_linear // cfg["kTileColCount"]
                tile_c = tile_linear % cfg["kTileColCount"]
                tile_row_start = tile_start(tile_r, cfg["kTileStrideRows"])
                tile_col_start = tile_start(tile_c, cfg["kTileStrideCols"])
                data_pos = lane_base + graph_iter * cfg["kOutputSampleElems"]
                for local_row in range(cfg["kOutputTileRows"]):
                    for local_col in range(cfg["kOutputTileCols"]):
                        value = values[data_pos]
                        data_pos += 1
                        global_row = tile_row_start + local_row
                        global_col = tile_col_start + local_col
                        if not (cfg["kOutputFirstRow"] <= global_row <= cfg["kOutputLastRow"]):
                            continue
                        if not (cfg["kOutputFirstCol"] <= global_col <= cfg["kOutputLastCol"]):
                            continue
                        out_r = global_row - cfg["kOutputFirstRow"]
                        out_c = global_col - cfg["kOutputFirstCol"]
                        out[out_r * output_cols + out_c] = value
        return out

    raise RuntimeError(
        f"AIE output has {len(values)} floats, expected {compact_elems} compact "
        f"or {expected_raw_elems} raw tile-stream or "
        f"{expected_lane_major_elems} lane-major raw tile-stream for "
        f"{iters} iteration(s)"
    )


def compare(got: list[float],
            expected: list[float],
            output_rows: int,
            cols: int,
            output_first_row: int,
            output_first_col: int,
            atol: float,
            rtol: float) -> bool:
    max_abs = -1.0
    max_i = -1
    bad = 0
    compared = 0

    for out_r in range(output_rows):
        for c in range(cols):
            i = out_r * cols + c
            g = got[i]
            e = expected[i]
            diff = abs(g - e)
            compared += 1
            if diff > max_abs:
                max_abs = diff
                max_i = i
            if diff > atol + rtol * abs(e):
                bad += 1

    out_r = max_i // cols
    c = max_i % cols
    print(f"compared pixels : {compared}")
    print(
        f"max abs diff    : {max_abs:.9g} at compact output ({out_r}, {c}), "
        f"source ({out_r + output_first_row}, {c + output_first_col})"
    )
    print(f"python gold     : {expected[max_i]:.9g}")
    print(f"aie output      : {got[max_i]:.9g}")
    print(f"bad pixels      : {bad}")
    if bad:
        print("compare result  : FAIL")
        return False
    print("compare result  : PASS")
    return True


def find_default_aie_output(base: Path) -> Path | None:
    for name in ("aie_j_next.txt", "aiesim_j_next.txt", "plio_ours_j_next.txt"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def is_raw_tile_output(path: Path) -> bool:
    return path.name in {"aiesim_j_next.txt", "plio_ours_j_next.txt"}


def main() -> int:
    base = Path(__file__).resolve().parent
    cfg, lambda_default, bypass = read_config()
    rows = cfg["kRows"]
    cols = cfg["kCols"]
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]

    parser = argparse.ArgumentParser(
        description="Generate a tile-stream SRAD Python gold file and compare AIE output."
    )
    parser.add_argument("--input", type=Path, default=base / "plio_ours_j.txt")
    parser.add_argument(
        "--gold-output",
        type=Path,
        default=base / "gold_srad.txt",
    )
    parser.add_argument(
        "--gold-tile-output",
        type=Path,
        default=base / "gold_srad_tile.txt",
        help="Write Python gold in raw tile-major PLIO64 order for sim diffing.",
    )
    parser.add_argument("--aie-output", type=Path, default=None)
    parser.add_argument(
        "--raw-tile-stream",
        action="store_true",
        help="Treat --aie-output as tile-major raw AIE PLIO output.",
    )
    parser.add_argument("--lambda-file", type=Path, default=base / "lambda.txt")
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument("--iters", type=int, default=cfg["kDefaultIterations"])
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()
    if args.iters < 1:
        raise RuntimeError("--iters must be positive")

    image = read_input_image(args.input, base, rows, cols)
    lam = (
        args.lambda_value
        if args.lambda_value is not None
        else read_scalar(args.lambda_file, lambda_default)
    )
    gold, q0_values, gold_tile = srad_multi_iteration(image, cfg, lam, bypass, args.iters)
    write_plio64_stream(args.gold_output, gold)
    write_plio64_stream(args.gold_tile_output, gold_tile)

    print(f"input          : {args.input if args.input.exists() else base / 'plio_ours_j.txt'}")
    print(f"gold output    : {args.gold_output}")
    print(f"gold tile raw  : {args.gold_tile_output}")
    print(f"image size     : {rows}x{cols}")
    print(f"iterations     : {args.iters}")
    print(
        f"tile schedule  : {cfg['kTileRowCount']}x{cfg['kTileColCount']} tiles, "
        f"output {cfg['kOutputTileRows']}x{cfg['kOutputTileCols']}, "
        f"raw output floats/tile {cfg['kOutputSampleElems']}"
    )
    print(f"graph firings  : {cfg['kGraphRunIterations'] * args.iters}")
    print(f"compact output : {output_rows}x{output_cols}")
    print(f"source rows    : {cfg['kOutputFirstRow']}..{cfg['kOutputLastRow']}")
    print(f"source cols    : {cfg['kOutputFirstCol']}..{cfg['kOutputLastCol']}")
    print(f"lambda         : {lam:.9g}")
    print(f"q0sqr first    : {q0_values[0]:.9g}")
    print(f"q0sqr last     : {q0_values[-1]:.9g}")
    print("compare region : all compact valid pixels")

    aie_output = args.aie_output or find_default_aie_output(base)
    if aie_output is None:
        print("compare        : skipped, no AIE output file was found")
        return 0

    got = normalize_aie_output(
        read_floats(aie_output),
        cfg,
        args.iters,
        force_raw_tile_stream=args.raw_tile_stream or is_raw_tile_output(aie_output),
    )
    print(f"aie output     : {aie_output}")
    return 0 if compare(
        got,
        gold,
        output_rows,
        output_cols,
        cfg["kOutputFirstRow"],
        cfg["kOutputFirstCol"],
        args.atol,
        args.rtol,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
