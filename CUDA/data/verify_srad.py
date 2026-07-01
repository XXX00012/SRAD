#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import gen_case


OUTPUT_PAIRS = (
    ("gpu_prepare_sums.txt", "gold_gpu_prepare_sums.txt"),
    ("gpu_prepare_sums2.txt", "gold_gpu_prepare_sums2.txt"),
    ("gpu_reduce_partial.txt", "gold_gpu_reduce_partial.txt"),
    ("gpu_srad_dN.txt", "gold_gpu_srad_dN.txt"),
    ("gpu_srad_dS.txt", "gold_gpu_srad_dS.txt"),
    ("gpu_srad_dW.txt", "gold_gpu_srad_dW.txt"),
    ("gpu_srad_dE.txt", "gold_gpu_srad_dE.txt"),
    ("gpu_srad_c.txt", "gold_gpu_srad_c.txt"),
    ("gpu_srad2_i_next.txt", "gold_gpu_srad2_i_next.txt"),
)


def read_config_ints(text: str) -> dict[str, int]:
    return gen_case.read_config_ints(text)


def read_config() -> tuple[dict[str, int], float]:
    return gen_case.read_config()


def read_floats(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(path)
    values: list[float] = []
    for token in path.read_text(encoding="utf-8").split():
        try:
            values.append(float(token))
        except ValueError:
            # AIE simulator output may include timestamps or labels depending on flow.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*[:=]?", token):
                continue
            raise
    return values


def resolve_output_path(path: Path) -> Path:
    if path.exists():
        return path

    root = path.parent.parent if path.parent.name == "data" else Path.cwd()
    search_roots = [
        root / "x86simulator_output",
        root / "aiesimulator_output",
        root / "Work" / "x86simulator_output",
        root / "Work" / "aiesimulator_output",
    ]
    candidates: list[Path] = []
    checked = [path]
    for search_root in search_roots:
        checked.append(search_root / "data" / path.name)
        checked.append(search_root / path.name)
        if search_root.exists():
            candidates.extend(search_root.rglob(path.name))

    for candidate in checked[1:] + candidates:
        if candidate.is_file():
            return candidate

    checked_text = "\n  ".join(str(candidate) for candidate in checked)
    raise FileNotFoundError(
        "AIE output not found. Checked:\n  "
        f"{checked_text}\n"
        "The simulator may have failed before writing the output PLIO."
    )


def compare_vectors(name: str,
                    got: list[float],
                    gold: list[float],
                    tol: float) -> bool:
    if len(got) != len(gold):
        raise RuntimeError(f"{name}: got {len(got)} floats, expected {len(gold)}")

    max_abs = 0.0
    max_rel = 0.0
    max_index = -1
    mismatches = 0
    for index, (got_value, gold_value) in enumerate(zip(got, gold)):
        abs_err = abs(got_value - gold_value)
        rel_err = abs_err / max(1.0, abs(gold_value))
        if abs_err > max_abs:
            max_abs = abs_err
            max_rel = rel_err
            max_index = index
        if abs_err > tol and rel_err > tol:
            mismatches += 1

    print(f"{name}: compared={len(gold)} max_abs={max_abs:.9g} max_rel={max_rel:.9g} mismatches={mismatches}")
    if max_index >= 0 and mismatches:
        print(
            f"{name}: max_index={max_index} "
            f"got={got[max_index]:.9g} gold={gold[max_index]:.9g}"
        )
    return mismatches == 0


def compare_all(base: Path, tol: float) -> bool:
    ok = True
    for got_name, gold_name in OUTPUT_PAIRS:
        got_path = resolve_output_path(base / got_name)
        gold_path = base / gold_name
        got = read_floats(got_path)
        gold = read_floats(gold_path)
        ok = compare_vectors(got_name, got, gold, tol) and ok
    print(f"compare: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Verify CUDA-style one-block AIE SRAD PLIO outputs."
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        cfg, lambda_default = read_config()
        print(f"block elems : {cfg['kCudaBlockElems']}")
        print(f"lambda      : {lambda_default:.9g}")
        print("gold files  : expected from data/gen_case.py --sim")

    if not args.compare:
        return 0
    return 0 if compare_all(base, args.tol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
