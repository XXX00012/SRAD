#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
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
        "kSimRows",
        "kRowDataElems",
        "kRowPhysElems",
        "kQ0PadIndex",
        "kStatSumPadIndex",
        "kStatSum2PadIndex",
        "kFlushRows",
        "kCenterRowLag",
        "kRowsPerIterSim",
        "kSradIterations",
        "kSimIterations",
        "kDefaultIterations",
        "kSimInvalidRows",
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


def resolve_aie_output_path(path: Path) -> Path:
    if path.exists():
        return path

    project_root = path.parent.parent if path.parent.name == "data" else Path.cwd()
    search_roots = [
        project_root / "x86simulator_output",
        project_root / "aiesimulator_output",
        project_root / "Work" / "x86simulator_output",
        project_root / "Work" / "aiesimulator_output",
    ]
    checked = [path]
    for root in search_roots:
        checked.append(root / "data" / path.name)
        checked.append(root / path.name)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in checked[1:]:
        if candidate not in seen:
            seen.add(candidate)
            if candidate.is_file():
                candidates.append(candidate)

    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob(path.name):
            if candidate not in seen:
                seen.add(candidate)
                if candidate.is_file():
                    candidates.append(candidate)

    if candidates:
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)

    checked_text = "\n  ".join(str(candidate) for candidate in checked)
    raise FileNotFoundError(
        "AIE output not found. Checked:\n  "
        f"{checked_text}\n"
        "The simulator may have failed before writing the output PLIO."
    )


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


def make_default_image(rows: int, cols: int) -> list[float]:
    image: list[float] = []
    for r in range(rows):
        for c in range(cols):
            image.append(
                1.0
                + 0.003 * r
                + 0.002 * c
                + 0.05 * math.sin(0.31 * r) * math.cos(0.19 * c)
            )
    return image


def read_input_image(path: Path, rows: int, cols: int) -> list[float]:
    pixels = rows * cols
    if path.exists():
        image = read_floats(path)
        if len(image) != pixels:
            raise RuntimeError(f"{path} has {len(image)} floats, expected {pixels}")
        return image
    return make_default_image(rows, cols)


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


def srad_next(image: list[float],
              rows: int,
              cols: int,
              lam: float,
              q0sqr: float,
              bypass: bool) -> list[float]:
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
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
            out[idx(r, c, cols)] = jc + 0.25 * lam * divergence
    return out


def make_output_row(image: list[float],
                    row_idx: int,
                    cfg: dict[str, int]) -> list[float]:
    cols = cfg["kCols"]
    row_data = cfg["kRowDataElems"]
    row_phys = cfg["kRowPhysElems"]
    row = list(image[row_idx * cols : row_idx * cols + row_data])
    row += [0.0] * (row_phys - row_data)
    row_sum = sum(row[:row_data])
    row_sum2 = sum(v * v for v in row[:row_data])
    row[cfg["kStatSumPadIndex"]] = row_sum
    row[cfg["kStatSum2PadIndex"]] = row_sum2
    return row


def make_rowstream_gold(image: list[float],
                        n_stream_rows: int,
                        n_valid_rows: int,
                        cfg: dict[str, int]) -> list[float]:
    row_phys = cfg["kRowPhysElems"]
    center_lag = cfg["kCenterRowLag"]
    invalid_rows = cfg["kSimInvalidRows"]
    out: list[float] = []
    for out_r in range(n_stream_rows):
        source_r = out_r - center_lag
        if out_r < invalid_rows or source_r < 0 or source_r >= n_valid_rows:
            out.extend([0.0] * row_phys)
        else:
            out.extend(make_output_row(image, source_r, cfg))
    return out


def srad_multi_iteration(image: list[float],
                         cfg: dict[str, int],
                         lam: float,
                         bypass: bool,
                         iters: int) -> tuple[list[float], list[float], list[float]]:
    sim_rows = cfg["kSimRows"]
    cols = cfg["kCols"]
    cur = image[: sim_rows * cols]
    q0_values: list[float] = []
    raw_all_iters: list[float] = []

    for _ in range(iters):
        q0sqr = compute_q0sqr(cur)
        q0_values.append(q0sqr)
        cur = srad_next(
            cur,
            sim_rows,
            cols,
            lam,
            q0sqr,
            bypass,
        )
        raw_all_iters.extend(
            make_rowstream_gold(cur, cfg["kRowsPerIterSim"], sim_rows, cfg)
        )

    return cur, q0_values, raw_all_iters


def compare_rowstream(got: list[float],
                      gold: list[float],
                      cfg: dict[str, int],
                      iters: int,
                      tol: float) -> bool:
    row_phys = cfg["kRowPhysElems"]
    row_data = cfg["kRowDataElems"]
    rows_per_iter = cfg["kRowsPerIterSim"]
    invalid_rows = cfg["kSimInvalidRows"]
    expected = rows_per_iter * iters * row_phys

    if len(got) != expected:
        raise RuntimeError(
            f"AIE output has {len(got)} floats, expected {expected} "
            f"for {iters} iteration(s)"
        )
    if len(gold) != expected:
        raise RuntimeError(
            f"gold rowstream has {len(gold)} floats, expected {expected}"
        )

    max_abs = 0.0
    max_i = -1
    mismatch_count = 0
    compared_count = 0

    for iter_idx in range(iters):
        iter_base = iter_idx * rows_per_iter * row_phys
        for row in range(rows_per_iter):
            if row < invalid_rows:
                continue
            row_base = iter_base + row * row_phys
            for col in range(row_phys):
                i = row_base + col
                abs_err = abs(got[i] - gold[i])
                if abs_err > max_abs:
                    max_abs = abs_err
                    max_i = i
                if abs_err > tol:
                    mismatch_count += 1
                compared_count += 1

    print(f"aie compared rows : {rows_per_iter - invalid_rows} per iter")
    print(f"compared floats   : {compared_count}")
    print(f"max abs diff      : {max_abs:.9g}")
    if max_i >= 0:
        iter_span = rows_per_iter * row_phys
        iter_idx = max_i // iter_span
        row = (max_i % iter_span) // row_phys
        col = max_i % row_phys
        print(f"max diff location : iter {iter_idx}, row {row}, col {col}")
        print(f"aie output        : {got[max_i]:.9g}")
        print(f"python gold       : {gold[max_i]:.9g}")
    print(f"mismatch count    : {mismatch_count}")
    return mismatch_count == 0


def main() -> int:
    base = Path(__file__).resolve().parent
    cfg, lambda_default, bypass = read_config()

    parser = argparse.ArgumentParser(
        description="Generate row-stream SRAD Python gold for AIE sim diffing."
    )
    parser.add_argument("--input", type=Path, default=base / "input_image_sim.txt")
    parser.add_argument(
        "--gold-output",
        type=Path,
        default=base / "gold_srad_rowstream.txt",
    )
    parser.add_argument(
        "--gold-final-output",
        type=Path,
        default=base / "gold_srad.txt",
    )
    parser.add_argument("--aie-output", type=Path, default=base / "aiesim_j_next.txt")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--iters", type=int, default=cfg["kSimIterations"])
    args = parser.parse_args()
    if args.iters < 1:
        raise RuntimeError("--iters must be positive")

    image = read_input_image(args.input, cfg["kRows"], cfg["kCols"])
    gold_final, q0_values, gold_stream = srad_multi_iteration(
        image, cfg, lambda_default, bypass, args.iters
    )
    write_plio64_stream(args.gold_output, gold_stream)
    write_matrix(args.gold_final_output, gold_final, cfg["kSimRows"], cfg["kCols"])

    if args.verbose:
        print(f"input          : {args.input}")
        print(f"gold rowstream : {args.gold_output}")
        print(f"gold final     : {args.gold_final_output}")
        print(f"image size     : {cfg['kRows']}x{cfg['kCols']}")
        print(
            f"rowstream      : sim rows/iter={cfg['kSimRows']}, "
            f"stream rows/iter={cfg['kRowsPerIterSim']}, "
            f"physical row={cfg['kRowPhysElems']}, "
            f"invalid rows/iter={cfg['kSimInvalidRows']}"
        )
        print(f"iterations     : {args.iters}")
        print(f"lambda         : {lambda_default:.9g}")
        for i, q0 in enumerate(q0_values):
            print(f"q0sqr iter{i}={q0:.9g}")
        print(f"q0sqr first    : {q0_values[0]:.9g}")
        print(f"q0sqr last     : {q0_values[-1]:.9g}")
        print(f"PLIO lines     : {len(gold_stream) // 2}")

    if not args.compare:
        if args.verbose:
            print("compare        : skipped")
        return 0

    aie_output = resolve_aie_output_path(args.aie_output)
    got = read_floats(aie_output)
    ok = compare_rowstream(got, gold_stream, cfg, args.iters, args.tol)
    print(f"aie output     : {aie_output}")
    print(f"compare        : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
