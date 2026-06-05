#!/usr/bin/env python3

import math
import re
import struct
from pathlib import Path

def read_config():
    cfg = Path(__file__).resolve().parents[1] / "aie" / "Config.h"
    text = cfg.read_text(encoding="utf-8")
    values = []
    for name in ("kRows", "kCols", "kLanes", "kBlockRows", "kBlockCols"):
        match = re.search(rf"constexpr\s+int\s+{name}\s*=\s*(\d+)\s*;", text)
        if match is None:
            raise RuntimeError(f"cannot read {name} from {cfg}")
        values.append(int(match.group(1)))
    return tuple(values)


ROWS, COLS, LANES, BLOCK_ROWS, BLOCK_COLS = read_config()
PIXELS = ROWS * COLS
LAMBDA = 0.5
OPENCL_REDUCTION_THREADS = 256
TILE_ROWS = (ROWS + BLOCK_ROWS - 1) // BLOCK_ROWS
TILE_COLS = (COLS + BLOCK_COLS - 1) // BLOCK_COLS
TILE_COUNT = TILE_ROWS * TILE_COLS
COEFF_INPUT_ROWS = ROWS
COEFF_INPUT_COLS = COLS
UPDATE_C_INPUT_ROWS = ROWS
UPDATE_C_INPUT_COLS = COLS
UPDATE_C_INPUT_ELEMS = PIXELS


def f32(v: float) -> float:
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def north_row(r: int) -> int:
    return 0 if r == 0 else r - 1


def south_row(r: int) -> int:
    return ROWS - 1 if r == ROWS - 1 else r + 1


def west_col(c: int) -> int:
    return 0 if c == 0 else c - 1


def east_col(c: int) -> int:
    return COLS - 1 if c == COLS - 1 else c + 1


def idx(r: int, c: int) -> int:
    return r + ROWS * c


def clamp_index(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def clamp01(v: float) -> float:
    return f32(max(0.0, min(1.0, v)))


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def largest_opencl_reduction_pow2(n: int) -> int:
    df = 1
    i = 2
    while i <= OPENCL_REDUCTION_THREADS:
        if n >= i:
            df = i
        i *= 2
    return df


def reduce_full_opencl_block(psum, psum2):
    i = 2
    while i <= OPENCL_REDUCTION_THREADS:
        for tx in range(i - 1, OPENCL_REDUCTION_THREADS, i):
            psum[tx] = f32(psum[tx] + psum[tx - i // 2])
            psum2[tx] = f32(psum2[tx] + psum2[tx - i // 2])
        i *= 2


def opencl_v0_reduce_stats(image):
    sums = [0.0] * PIXELS
    sums2 = [0.0] * PIXELS
    for opencl_idx, value in enumerate(image):
        v = f32(value)
        sums[opencl_idx] = v
        sums2[opencl_idx] = f32(v * v)

    no = PIXELS
    mul = 1
    grid_dim = ceil_div(no, OPENCL_REDUCTION_THREADS)

    while grid_dim != 0:
        nf = OPENCL_REDUCTION_THREADS - (
            grid_dim * OPENCL_REDUCTION_THREADS - no
        )
        for bx in range(grid_dim):
            psum = [0.0] * OPENCL_REDUCTION_THREADS
            psum2 = [0.0] * OPENCL_REDUCTION_THREADS
            for tx in range(OPENCL_REDUCTION_THREADS):
                ei = bx * OPENCL_REDUCTION_THREADS + tx
                if ei < no:
                    src = ei * mul
                    psum[tx] = sums[src]
                    psum2[tx] = sums2[src]

            dst = bx * mul * OPENCL_REDUCTION_THREADS
            if nf == OPENCL_REDUCTION_THREADS or bx != grid_dim - 1:
                reduce_full_opencl_block(psum, psum2)
                sums[dst] = psum[OPENCL_REDUCTION_THREADS - 1]
                sums2[dst] = psum2[OPENCL_REDUCTION_THREADS - 1]
            else:
                df = largest_opencl_reduction_pow2(nf)
                i = 2
                while i <= df:
                    for tx in range(i - 1, df, i):
                        psum[tx] = f32(psum[tx] + psum[tx - i // 2])
                        psum2[tx] = f32(psum2[tx] + psum2[tx - i // 2])
                    i *= 2

                tx = df - 1
                for i in range(
                    bx * OPENCL_REDUCTION_THREADS + df,
                    bx * OPENCL_REDUCTION_THREADS + nf,
                ):
                    psum[tx] = f32(psum[tx] + sums[i])
                    psum2[tx] = f32(psum2[tx] + sums2[i])
                sums[dst] = psum[tx]
                sums2[dst] = psum2[tx]

        no = grid_dim
        if grid_dim == 1:
            grid_dim = 0
        else:
            mul *= OPENCL_REDUCTION_THREADS
            grid_dim = ceil_div(no, OPENCL_REDUCTION_THREADS)

    return sums[0], sums2[0]


def compute_q0sqr(image):
    sum_j, sum2_j = opencl_v0_reduce_stats(image)
    mean = f32(sum_j / f32(PIXELS))
    mean2 = f32(mean * mean)
    variance = f32(f32(sum2_j / f32(PIXELS)) - mean2)
    return f32(variance / mean2)


def compute_c(jc, d_n, d_s, d_w, d_e, q0sqr):
    g2_sum = f32(f32(f32(d_n * d_n) + f32(d_s * d_s)) + f32(d_w * d_w))
    g2_sum = f32(g2_sum + f32(d_e * d_e))
    g2 = f32(g2_sum / f32(jc * jc))
    lap = f32(f32(f32(f32(d_n + d_s) + d_w) + d_e) / jc)
    num = f32(f32(0.5 * g2) - f32(f32(1.0 / 16.0) * f32(lap * lap)))
    den = f32(1.0 + f32(0.25 * lap))
    qsqr = f32(num / f32(den * den))
    den = f32(f32(qsqr - q0sqr) / f32(q0sqr * f32(1.0 + q0sqr)))
    c = f32(1.0 / f32(1.0 + den))
    return clamp01(c)


def srad_one_iteration(image, q0sqr, lam):
    c_plane = [0.0] * PIXELS
    dn_plane = [0.0] * PIXELS
    ds_plane = [0.0] * PIXELS
    dw_plane = [0.0] * PIXELS
    de_plane = [0.0] * PIXELS

    for r in range(ROWS):
        for c in range(COLS):
            p = idx(r, c)
            jc = f32(image[p])
            d_n = f32(image[idx(north_row(r), c)] - jc)
            d_s = f32(image[idx(south_row(r), c)] - jc)
            d_w = f32(image[idx(r, west_col(c))] - jc)
            d_e = f32(image[idx(r, east_col(c))] - jc)
            dn_plane[p] = d_n
            ds_plane[p] = d_s
            dw_plane[p] = d_w
            de_plane[p] = d_e
            c_plane[p] = compute_c(jc, d_n, d_s, d_w, d_e, q0sqr)

    out = [0.0] * PIXELS
    for r in range(ROWS):
        for c in range(COLS):
            p = idx(r, c)
            d_val = f32(c_plane[p] * dn_plane[p])
            d_val = f32(d_val + f32(c_plane[idx(south_row(r), c)] * ds_plane[p]))
            d_val = f32(d_val + f32(c_plane[p] * dw_plane[p]))
            d_val = f32(d_val + f32(c_plane[idx(r, east_col(c))] * de_plane[p]))
            out[p] = f32(image[p] + f32(f32(0.25 * lam) * d_val))

    return out, c_plane, dn_plane, ds_plane, dw_plane, de_plane


def write_matrix(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for r in range(ROWS):
            row = [data[idx(r, c)] for c in range(COLS)]
            f.write(" ".join(f"{v:.9g}" for v in row))
            f.write("\n")


def write_stream(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for v in data:
            f.write(f"{v:.9g}\n")


def pack_coeff_image_stream(image):
    out = []
    for _ in range(TILE_COUNT):
        out.extend(image)
    return out


def pack_plain_tile_stream(image):
    out = []
    for tr in range(TILE_ROWS):
        for tc in range(TILE_COLS):
            row0 = tr * BLOCK_ROWS
            col0 = tc * BLOCK_COLS
            for c in range(BLOCK_COLS):
                for r in range(BLOCK_ROWS):
                    gr = clamp_index(row0 + r, 0, ROWS - 1)
                    gc = clamp_index(col0 + c, 0, COLS - 1)
                    out.append(image[idx(gr, gc)])
    return out


def pack_update_c_stream(c_plane):
    out = []
    for _ in range(TILE_COUNT):
        out.extend(c_plane)
    return out


def scalar_packet_stream(value):
    out = []
    for _ in range(TILE_COUNT):
        out.append(f32(value))
        out.extend([0.0] * (LANES - 1))
    return out


def main():
    base = Path(__file__).resolve().parent
    image = [0.0] * PIXELS
    for r in range(ROWS):
        for c in range(COLS):
            v = (
                1.0
                + 0.003 * r
                + 0.002 * c
                + 0.05 * math.sin(0.31 * r) * math.cos(0.19 * c)
            )
            image[idx(r, c)] = f32(v)

    q0sqr = compute_q0sqr(image)
    sums_plane = list(image)
    sums2_plane = [f32(v * v) for v in image]
    golden, c_plane, dn_plane, ds_plane, dw_plane, de_plane = srad_one_iteration(
        image, q0sqr, LAMBDA
    )

    write_matrix(base / "input_32x32.txt", image)
    write_matrix(base / "golden_32x32.txt", golden)
    write_matrix(base / "ref_d_sums.txt", sums_plane)
    write_matrix(base / "ref_d_sums2.txt", sums2_plane)
    write_matrix(base / "ref_d_c.txt", c_plane)
    write_matrix(base / "ref_d_dN.txt", dn_plane)
    write_matrix(base / "ref_d_dS.txt", ds_plane)
    write_matrix(base / "ref_d_dW.txt", dw_plane)
    write_matrix(base / "ref_d_dE.txt", de_plane)

    write_stream(base / "plio_prepare_j.txt", image)
    write_stream(base / "plio_prepare_sums.txt", sums_plane)
    write_stream(base / "plio_prepare_sums2.txt", sums2_plane)
    write_stream(base / "plio_coeff_j.txt", pack_coeff_image_stream(image))
    write_stream(base / "plio_update_j.txt", pack_plain_tile_stream(image))
    write_stream(base / "plio_runtime_coeff_q0sqr.txt", scalar_packet_stream(q0sqr))
    write_stream(base / "q0sqr.txt", scalar_packet_stream(q0sqr))
    write_stream(base / "plio_runtime_update_c.txt", pack_update_c_stream(c_plane))
    write_stream(base / "plio_runtime_update_dN.txt", pack_plain_tile_stream(dn_plane))
    write_stream(base / "plio_runtime_update_dS.txt", pack_plain_tile_stream(ds_plane))
    write_stream(base / "plio_runtime_update_dW.txt", pack_plain_tile_stream(dw_plane))
    write_stream(base / "plio_runtime_update_dE.txt", pack_plain_tile_stream(de_plane))
    write_stream(base / "plio_runtime_lambda.txt", scalar_packet_stream(LAMBDA))

    (base / "q0sqr_ref.txt").write_text(
        "\n".join(f"{q0sqr:.9g}" for _ in range(8)) + "\n",
        encoding="utf-8",
    )
    (base / "lambda.txt").write_text(f"{LAMBDA:.9g}\n", encoding="utf-8")
    print(f"generated {ROWS}x{COLS} OpenCL-v0-layout CUDA_PL SRAD case")
    print(f"q0sqr={q0sqr:.9g} lambda={LAMBDA:.9g}")


if __name__ == "__main__":
    main()
