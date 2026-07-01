#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "aie" / "Config.h"


def read_config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


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


def pick_int(values: dict[str, int], *names: str, default: int | None = None) -> int:
    for name in names:
        if name in values:
            return values[name]
    if default is not None:
        return default
    raise RuntimeError(f"cannot read {', '.join(names)} from {CONFIG_PATH}")


def parse_lambda(text: str) -> float:
    lam_match = re.search(
        r"constexpr\s+float\s+kLambdaDefault\s*=\s*([0-9eE+\-.]+)f?\s*;",
        text,
    )
    return float(lam_match.group(1)) if lam_match else 0.5


def read_board_config(text: str | None = None) -> tuple[dict[str, int], float]:
    if text is None:
        text = read_config_text()
    values = read_config_ints(text)
    rows = pick_int(values, "kBoardRows", "kRows")
    cols = pick_int(values, "kBoardCols", "kCols")
    pixels = values.get("kBoardPixels", values.get("kPixels", rows * cols))
    cfg = {
        "kBoardRows": rows,
        "kBoardCols": cols,
        "kBoardPixels": pixels,
    }
    return cfg, parse_lambda(text)


def read_config() -> tuple[dict[str, int], float]:
    text = read_config_text()
    values = read_config_ints(text)
    board_cfg, lambda_default = read_board_config(text)
    required = (
        "kCudaBlockElems",
        "kMetaPacketElems",
        "kReducePacketElems",
        "kCoeffInputElems",
        "kUpdateInputElems",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"cannot read {', '.join(missing)} from {CONFIG_PATH}")

    cfg = dict(values)
    cfg.update(board_cfg)
    return cfg, lambda_default


CFG: dict[str, int] = {}
LAMBDA_DEFAULT = 0.5
ROWS = 0
COLS = 0
PIXELS = 0
BLOCK = 0
META = 0


def apply_runtime_config(cfg: dict[str, int], lambda_default: float) -> None:
    global CFG, LAMBDA_DEFAULT, ROWS, COLS, PIXELS, BLOCK, META
    CFG = cfg
    LAMBDA_DEFAULT = lambda_default
    ROWS = cfg["kBoardRows"]
    COLS = cfg["kBoardCols"]
    PIXELS = cfg["kBoardPixels"]
    BLOCK = cfg.get("kCudaBlockElems", 0)
    META = cfg.get("kMetaPacketElems", 0)


GRAPH_INPUT_FILES = (
    "gpu_prepare_i.txt",
    "gpu_reduce_packet.txt",
    "gpu_srad_neighbors.txt",
    "gpu_srad_q0.txt",
    "gpu_srad2_update.txt",
    "gpu_srad2_meta.txt",
)

GRAPH_OUTPUT_FILES = (
    "gpu_prepare_sums.txt",
    "gpu_prepare_sums2.txt",
    "gpu_reduce_partial.txt",
    "gpu_srad_dN.txt",
    "gpu_srad_dS.txt",
    "gpu_srad_dW.txt",
    "gpu_srad_dE.txt",
    "gpu_srad_c.txt",
    "gpu_srad2_i_next.txt",
)

STALE_FILES = (
    "aie_j_next.txt",
    "aiesim_j_next.txt",
    "aiesimoutput.txt",
    "gold_srad_rowstream.txt",
    "input_image_sim.txt",
    "plio_ours_j.txt",
    "plio_ours_j_next.txt",
    "plio_ours_j_tile.txt",
    "q0sqr.txt",
    "q0sqr_ref.txt",
)


def default_pixel(row: int, col: int) -> float:
    return (
        1.0
        + 0.003 * row
        + 0.002 * col
        + 0.05 * math.sin(0.31 * row) * math.cos(0.19 * col)
    )


def pixel_at_linear(index: int) -> float:
    return default_pixel(index // COLS, index % COLS)


def sample_image(index: int, row_delta: int = 0, col_delta: int = 0) -> float:
    row = index // COLS + row_delta
    col = index % COLS + col_delta
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return 0.0
    return default_pixel(row, col)


def write_matrix(path: Path, rows: int, cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in range(rows):
            values = (f"{default_pixel(row, col):.9g}" for col in range(cols))
            fout.write(" ".join(values) + "\n")


def write_plio64_stream(path: Path, values: list[float]) -> None:
    if len(values) % 2 != 0:
        raise ValueError(f"{path} requires an even float count for plio_64_bits")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for i in range(0, len(values), 2):
            fout.write(f"{values[i]:.9g} {values[i + 1]:.9g}\n")


def compute_q0sqr(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = (sum(v * v for v in values) / len(values)) - (mean * mean)
    return variance / (mean * mean) if mean != 0.0 else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def coeff_from_neighbors(jc: float,
                         jn: float,
                         js: float,
                         jw: float,
                         je: float,
                         q0sqr: float) -> tuple[float, float, float, float, float]:
    d_n = jn - jc
    d_s = js - jc
    d_w = jw - jc
    d_e = je - jc
    if jc == 0.0 or q0sqr == 0.0:
        return d_n, d_s, d_w, d_e, 1.0

    g2 = ((d_n * d_n) + (d_s * d_s) + (d_w * d_w) + (d_e * d_e)) / (jc * jc)
    lap = (d_n + d_s + d_w + d_e) / jc
    num = (0.5 * g2) - ((1.0 / 16.0) * lap * lap)
    den0 = 1.0 + (0.25 * lap)
    qsqr = num / (den0 * den0)
    den1 = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr))
    return d_n, d_s, d_w, d_e, clamp01(1.0 / (1.0 + den1))


def coeff_at(index: int, q0sqr: float) -> tuple[float, float, float, float, float]:
    return coeff_from_neighbors(
        sample_image(index),
        sample_image(index, row_delta=-1),
        sample_image(index, row_delta=1),
        sample_image(index, col_delta=-1),
        sample_image(index, col_delta=1),
        q0sqr,
    )


def make_prepare_input(block_index: int = 0) -> list[float]:
    base = block_index * BLOCK
    return [pixel_at_linear(base + i) for i in range(BLOCK)]


def make_reduce_packet(sums: list[float], sums2: list[float]) -> list[float]:
    packet = [0.0] * META
    packet[0] = float(len(sums))
    for sum_value, sum2_value in zip(sums, sums2):
        packet.append(sum_value)
        packet.append(sum2_value)
    return packet


def make_neighbors(block_index: int, q0sqr: float) -> tuple[list[float], dict[str, list[float]]]:
    base = block_index * BLOCK
    planes = {
        "jc": [],
        "jn": [],
        "js": [],
        "jw": [],
        "je": [],
        "dN": [],
        "dS": [],
        "dW": [],
        "dE": [],
        "c": [],
    }

    for i in range(BLOCK):
        index = base + i
        jc = sample_image(index)
        jn = sample_image(index, row_delta=-1)
        js = sample_image(index, row_delta=1)
        jw = sample_image(index, col_delta=-1)
        je = sample_image(index, col_delta=1)
        d_n, d_s, d_w, d_e, coeff = coeff_from_neighbors(jc, jn, js, jw, je, q0sqr)
        planes["jc"].append(jc)
        planes["jn"].append(jn)
        planes["js"].append(js)
        planes["jw"].append(jw)
        planes["je"].append(je)
        planes["dN"].append(d_n)
        planes["dS"].append(d_s)
        planes["dW"].append(d_w)
        planes["dE"].append(d_e)
        planes["c"].append(coeff)

    neighbors = planes["jc"] + planes["jn"] + planes["js"] + planes["jw"] + planes["je"]
    return neighbors, planes


def make_update_input(block_index: int,
                      q0sqr: float,
                      planes: dict[str, list[float]]) -> tuple[list[float], list[float]]:
    base = block_index * BLOCK
    image = [sample_image(base + i) for i in range(BLOCK)]
    c_s: list[float] = []
    c_e: list[float] = []
    for i in range(BLOCK):
        index = base + i
        row = index // COLS
        col = index % COLS
        c_s.append(coeff_at(index + COLS, q0sqr)[4] if row + 1 < ROWS else 0.0)
        c_e.append(coeff_at(index + 1, q0sqr)[4] if col + 1 < COLS else 0.0)

    update = (
        image
        + planes["dN"]
        + planes["dS"]
        + planes["dW"]
        + planes["dE"]
        + planes["c"]
        + c_s
        + c_e
    )

    expected: list[float] = []
    scale = 0.25 * LAMBDA_DEFAULT
    for i, value in enumerate(image):
        divergence = (
            planes["c"][i] * planes["dN"][i]
            + c_s[i] * planes["dS"][i]
            + planes["c"][i] * planes["dW"][i]
            + c_e[i] * planes["dE"][i]
        )
        expected.append(value + scale * divergence)

    return update, expected


def cleanup(base: Path) -> None:
    for name in STALE_FILES + GRAPH_OUTPUT_FILES:
        path = base / name
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass


def generate_board_input(base: Path, verbose: bool) -> None:
    write_matrix(base / "input_image.txt", ROWS, COLS)
    if verbose:
        print(f"generated board input {ROWS}x{COLS} ({PIXELS} floats)")


def generate_sim_inputs(base: Path, verbose: bool) -> None:
    cleanup(base)
    block_index = 0
    prepare_i = make_prepare_input(block_index)
    sums = prepare_i[:]
    sums2 = [value * value for value in prepare_i]
    q0sqr = compute_q0sqr(prepare_i)
    reduce_packet = make_reduce_packet(sums, sums2)
    neighbors, coeff_planes = make_neighbors(block_index, q0sqr)
    update_packet, update_expected = make_update_input(block_index, q0sqr, coeff_planes)

    write_plio64_stream(base / "gpu_prepare_i.txt", prepare_i)
    write_plio64_stream(base / "gpu_reduce_packet.txt", reduce_packet)
    write_plio64_stream(base / "gpu_srad_neighbors.txt", neighbors)
    write_plio64_stream(base / "gpu_srad_q0.txt", [q0sqr] + [0.0] * (META - 1))
    write_plio64_stream(base / "gpu_srad2_update.txt", update_packet)
    write_plio64_stream(
        base / "gpu_srad2_meta.txt",
        [LAMBDA_DEFAULT] + [0.0] * (META - 1),
    )

    # Golden files are for the Python verifier only; AIE writes the gpu_* output names.
    write_plio64_stream(base / "gold_gpu_prepare_sums.txt", sums)
    write_plio64_stream(base / "gold_gpu_prepare_sums2.txt", sums2)
    write_plio64_stream(base / "gold_gpu_reduce_partial.txt", [sum(sums), sum(sums2)])
    write_plio64_stream(base / "gold_gpu_srad_dN.txt", coeff_planes["dN"])
    write_plio64_stream(base / "gold_gpu_srad_dS.txt", coeff_planes["dS"])
    write_plio64_stream(base / "gold_gpu_srad_dW.txt", coeff_planes["dW"])
    write_plio64_stream(base / "gold_gpu_srad_dE.txt", coeff_planes["dE"])
    write_plio64_stream(base / "gold_gpu_srad_c.txt", coeff_planes["c"])
    write_plio64_stream(base / "gold_gpu_srad2_i_next.txt", update_expected)

    if verbose:
        print(f"generated one-block AIE CUDA-style sim case, block={BLOCK}")
        print(f"q0sqr={q0sqr:.9g}, lambda={LAMBDA_DEFAULT:.9g}")
        print(f"reduce packet floats={len(reduce_packet)}")
        print(f"coeff input floats={len(neighbors)}")
        print(f"update input floats={len(update_packet)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SRAD CUDA-style AIE data.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="generate the one-block PLIO files used by x86/aie simulation",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    if args.sim:
        cfg, lambda_default = read_config()
    else:
        cfg, lambda_default = read_board_config()
    apply_runtime_config(cfg, lambda_default)

    if args.sim:
        generate_sim_inputs(base, args.verbose)
    else:
        generate_board_input(base, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
