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
        "kOutputTileDataElems",
        "kOutputStatElems",
        "kOutputSampleElems",
        "kParallelLanes",
        "kOutputLanesPerPlio",
        "kOutputPlioGroups",
        "kTileRowCount",
        "kTileColCount",
        "kTotalTileCount",
        "kTileStrideRows",
        "kTileStrideCols",
        "kGraphRunIterations",
        "kSradIterations",
        "kDefaultIterations",
        "kOutputFirstRow",
        "kOutputLastRow",
        "kOutputFirstCol",
        "kOutputLastCol",
        "kOutputRows",
        "kOutputCols",
        "kCompareRows",
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


def tile_start(tile_idx: int, stride: int) -> int:
    return tile_idx * stride


def append_tile_sample(raw: list[float],
                       compact: list[float],
                       tile_linear: int,
                       cfg: dict[str, int]) -> None:
    if tile_linear >= cfg["kTotalTileCount"]:
        raw.extend([0.0] * cfg["kOutputSampleElems"])
        return

    output_cols = cfg["kOutputCols"]
    tile_r = tile_linear // cfg["kTileColCount"]
    tile_c = tile_linear % cfg["kTileColCount"]
    tile_row_start = tile_start(tile_r, cfg["kTileStrideRows"])
    tile_col_start = tile_start(tile_c, cfg["kTileStrideCols"])
    tile_sum = 0.0
    tile_sum2 = 0.0

    for local_row in range(cfg["kOutputTileRows"]):
        global_row = tile_row_start + local_row
        for local_col in range(cfg["kOutputTileCols"]):
            global_col = tile_col_start + local_col
            if not (cfg["kOutputFirstRow"] <= global_row <= cfg["kOutputLastRow"]):
                value = 0.0
            elif not (cfg["kOutputFirstCol"] <= global_col <= cfg["kOutputLastCol"]):
                value = 0.0
            else:
                out_r = global_row - cfg["kOutputFirstRow"]
                out_c = global_col - cfg["kOutputFirstCol"]
                value = compact[out_r * output_cols + out_c]
                tile_sum += value
                tile_sum2 += value * value
            raw.append(value)

    raw.append(tile_sum)
    raw.append(tile_sum2)


def compact_to_group_streams(compact: list[float],
                             cfg: dict[str, int]) -> list[list[float]]:
    groups: list[list[float]] = [
        [] for _ in range(cfg["kOutputPlioGroups"])
    ]
    for tile_iter in range(cfg["kGraphRunIterations"]):
        for group in range(cfg["kOutputPlioGroups"]):
            for local in range(cfg["kOutputLanesPerPlio"]):
                lane = group * cfg["kOutputLanesPerPlio"] + local
                tile_linear = tile_iter * cfg["kParallelLanes"] + lane
                append_tile_sample(groups[group], compact, tile_linear, cfg)
    return groups


def srad_multi_iteration(image: list[float],
                         cfg: dict[str, int],
                         lam: float,
                         bypass: bool,
                         iters: int) -> tuple[list[float], list[float], list[list[float]]]:
    cur = image
    q0_values: list[float] = []
    group_streams: list[list[float]] = [
        [] for _ in range(cfg["kOutputPlioGroups"])
    ]

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
        iter_groups = compact_to_group_streams(cur, cfg)
        for group, stream in enumerate(iter_groups):
            group_streams[group].extend(stream)

    return cur, q0_values, group_streams


def store_tile_output(out: list[float],
                      values: list[float],
                      pos: int,
                      tile_linear: int,
                      cfg: dict[str, int]) -> int:
    tile_base = pos
    if tile_linear >= cfg["kTotalTileCount"]:
        return tile_base + cfg["kOutputSampleElems"]

    tile_r = tile_linear // cfg["kTileColCount"]
    tile_c = tile_linear % cfg["kTileColCount"]
    tile_row_start = tile_start(tile_r, cfg["kTileStrideRows"])
    tile_col_start = tile_start(tile_c, cfg["kTileStrideCols"])

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
            out[out_r * cfg["kOutputCols"] + out_c] = value
    return tile_base + cfg["kOutputSampleElems"]


def normalize_group_values(values: list[float],
                           cfg: dict[str, int],
                           iters: int,
                           group: int,
                           out: list[float]) -> None:
    group_one_iter_elems = (
        cfg["kGraphRunIterations"]
        * cfg["kOutputLanesPerPlio"]
        * cfg["kOutputSampleElems"]
    )
    group_expected_elems = group_one_iter_elems * iters
    if len(values) == group_one_iter_elems:
        iter_offset = 0
    elif len(values) == group_expected_elems:
        iter_offset = (iters - 1) * group_one_iter_elems
    else:
        raise RuntimeError(
            f"group {group} output has {len(values)} floats, expected "
            f"{group_one_iter_elems} for one iteration or "
            f"{group_expected_elems} for {iters} iteration(s)"
        )

    pos = iter_offset
    for tile_iter in range(cfg["kGraphRunIterations"]):
        for local in range(cfg["kOutputLanesPerPlio"]):
            lane = group * cfg["kOutputLanesPerPlio"] + local
            tile_linear = tile_iter * cfg["kParallelLanes"] + lane
            pos = store_tile_output(out, values, pos, tile_linear, cfg)


def merge_group_outputs(paths: list[Path],
                        cfg: dict[str, int],
                        iters: int) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    compact = [0.0] * (output_rows * output_cols)

    for group, path in enumerate(paths):
        values = read_floats(path)
        normalize_group_values(values, cfg, iters, group, compact)
    return compact


def normalize_lane_values(values: list[float],
                          cfg: dict[str, int],
                          iters: int,
                          lane: int,
                          out: list[float]) -> None:
    lane_one_iter_elems = cfg["kGraphRunIterations"] * cfg["kOutputSampleElems"]
    lane_expected_elems = lane_one_iter_elems * iters
    if len(values) == lane_one_iter_elems:
        iter_offset = 0
    elif len(values) == lane_expected_elems:
        iter_offset = (iters - 1) * lane_one_iter_elems
    else:
        raise RuntimeError(
            f"lane {lane} output has {len(values)} floats, expected "
            f"{lane_one_iter_elems} for one iteration or "
            f"{lane_expected_elems} for {iters} iteration(s)"
        )

    pos = iter_offset
    for tile_iter in range(cfg["kGraphRunIterations"]):
        tile_linear = tile_iter * cfg["kParallelLanes"] + lane
        pos = store_tile_output(out, values, pos, tile_linear, cfg)


def merge_lane_outputs(paths: list[Path],
                       cfg: dict[str, int],
                       iters: int) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    compact = [0.0] * (output_rows * output_cols)

    for lane, path in enumerate(paths):
        values = read_floats(path)
        normalize_lane_values(values, cfg, iters, lane, compact)
    return compact


def normalize_flat_group_streams(values: list[float],
                                 cfg: dict[str, int],
                                 iters: int) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    compact = [0.0] * (output_rows * output_cols)
    group_one_iter_elems = (
        cfg["kGraphRunIterations"]
        * cfg["kOutputLanesPerPlio"]
        * cfg["kOutputSampleElems"]
    )
    group_expected_elems = group_one_iter_elems * iters
    all_groups_elems = cfg["kOutputPlioGroups"] * group_expected_elems
    if len(values) != all_groups_elems:
        raise RuntimeError(
            f"flat grouped output has {len(values)} floats, expected "
            f"{all_groups_elems}"
        )

    for group in range(cfg["kOutputPlioGroups"]):
        begin = group * group_expected_elems
        end = begin + group_expected_elems
        normalize_group_values(values[begin:end], cfg, iters, group, compact)
    return compact


def group_from_output_path(path: Path) -> int | None:
    match = re.fullmatch(r"plio_ours_j_next_group_(\d+)\.txt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def lane_from_output_path(path: Path) -> int | None:
    match = re.fullmatch(r"plio_ours_j_next_(\d+)\.txt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def normalize_aie_output(values: list[float],
                         cfg: dict[str, int],
                         iters: int,
                         path: Path | None = None,
                         force_raw_tile_stream: bool = False) -> list[float]:
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]
    compact_elems = output_rows * output_cols
    if len(values) == compact_elems and not force_raw_tile_stream:
        return values

    compact = [0.0] * compact_elems
    if path is not None:
        group = group_from_output_path(path)
        if group is not None:
            normalize_group_values(values, cfg, iters, group, compact)
            return compact

        lane = lane_from_output_path(path)
        if lane is not None:
            normalize_lane_values(values, cfg, iters, lane, compact)
            return compact

    return normalize_flat_group_streams(values, cfg, iters)


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
    for name in ("aie_j_next.txt", "aiesim_j_next.txt"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def find_default_group_outputs(base: Path, cfg: dict[str, int]) -> list[Path] | None:
    paths = [
        base / f"plio_ours_j_next_group_{group}.txt"
        for group in range(cfg["kOutputPlioGroups"])
    ]
    if all(path.exists() for path in paths):
        return paths
    return None


def find_default_lane_outputs(base: Path, cfg: dict[str, int]) -> list[Path] | None:
    paths = [
        base / f"plio_ours_j_next_{lane}.txt"
        for lane in range(cfg["kParallelLanes"])
    ]
    if all(path.exists() for path in paths):
        return paths
    return None


def is_raw_tile_output(path: Path) -> bool:
    return (
        path.name in {"plio_ours_j_next.txt", "gold_srad_tile.txt"}
        or group_from_output_path(path) is not None
        or lane_from_output_path(path) is not None
    )


def main() -> int:
    base = Path(__file__).resolve().parent
    cfg, lambda_default, bypass = read_config()
    rows = cfg["kRows"]
    cols = cfg["kCols"]
    output_rows = cfg["kOutputRows"]
    output_cols = cfg["kOutputCols"]

    parser = argparse.ArgumentParser(
        description="Generate a multi-iteration SRAD Python gold file and compare AIE output."
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
        help="Write Python gold as concatenated group raw PLIO64 streams.",
    )
    parser.add_argument("--aie-output", type=Path, default=None)
    parser.add_argument(
        "--raw-tile-stream",
        action="store_true",
        help="Treat --aie-output as raw grouped AIE PLIO output.",
    )
    parser.add_argument("--lambda-file", type=Path, default=base / "lambda.txt")
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument("--iters", type=int, default=cfg["kDefaultIterations"])
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()
    if args.iters < 1 or args.iters > cfg["kSradIterations"]:
        raise RuntimeError(
            f"--iters must be in 1..{cfg['kSradIterations']} for this build"
        )

    image = read_input_image(args.input, base, rows, cols)
    lam = (
        args.lambda_value
        if args.lambda_value is not None
        else read_scalar(args.lambda_file, lambda_default)
    )
    gold, q0_values, group_gold = srad_multi_iteration(
        image, cfg, lam, bypass, args.iters
    )
    write_matrix(args.gold_output, gold, output_rows, output_cols)
    write_plio64_stream(
        args.gold_tile_output,
        [value for group in group_gold for value in group],
    )

    print(f"input          : {args.input if args.input.exists() else base / 'plio_ours_j.txt'}")
    print(f"gold output    : {args.gold_output}")
    print(f"gold grouped   : {args.gold_tile_output}")
    print(f"image size     : {rows}x{cols}")
    print(f"iterations     : {args.iters}")
    print(
        f"tile schedule  : {cfg['kTileRowCount']}x{cfg['kTileColCount']} tiles, "
        f"output {cfg['kOutputTileRows']}x{cfg['kOutputTileCols']}, "
        f"raw output floats/tile {cfg['kOutputSampleElems']}"
    )
    print(
        f"parallel layout : {cfg['kParallelLanes']} lanes, "
        f"{cfg['kOutputPlioGroups']} output groups, "
        f"pktmerge<{cfg['kOutputLanesPerPlio']}>"
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
    group_outputs = None
    lane_outputs = None
    if aie_output is None:
        group_outputs = find_default_group_outputs(base, cfg)
    if aie_output is None and group_outputs is None:
        lane_outputs = find_default_lane_outputs(base, cfg)
    if aie_output is None and group_outputs is None and lane_outputs is None:
        print("compare        : skipped, no AIE output file was found")
        return 0

    if group_outputs is not None:
        got = merge_group_outputs(group_outputs, cfg, args.iters)
        print(
            f"aie output     : {base / 'plio_ours_j_next_group_[0..N].txt'} "
            f"({cfg['kOutputPlioGroups']} group files)"
        )
    elif lane_outputs is not None:
        got = merge_lane_outputs(lane_outputs, cfg, args.iters)
        print(
            f"aie output     : {base / 'plio_ours_j_next_[0..N].txt'} "
            f"({cfg['kParallelLanes']} lane files)"
        )
    else:
        got = normalize_aie_output(
            read_floats(aie_output),
            cfg,
            args.iters,
            path=aie_output,
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
