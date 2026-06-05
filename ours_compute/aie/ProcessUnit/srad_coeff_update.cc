#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// Ours-compute K2: original SRAD coefficient generation, line-buffer
// c-neighbor alignment, divergence, and update.
//
// K2 per-pixel approximate arithmetic:
//   add/sub: about 8
//   mul: about 8
//   div: 5 per lane, implemented as five vdiv passes per v8float block
//   plus clamp and c-neighbor buffering
//
// DAG nodes:
//   g2/lap/qsqr/ratio/c generation using the original SRAD expression
//   c-neighbor alignment
//   divergence
//   update
//
// K2 consumes the K1 local mid buffer record:
//   unused, dN, dS, dW, dE, JC
//
// Original coefficient form:
//   A = dN*dN + dS*dS + dW*dW + dE*dE
//   B = dN + dS + dW + dE
//   g2   = A / (JC*JC)
//   lap  = B / JC
//   qsqr = (0.5*g2 - (1/16)*lap*lap) / (1 + 0.25*lap)^2
//   ratio = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr))
//   c = 1 / (1 + ratio)
//
// This deliberately keeps the five division points so this variant isolates
// the benefit of the algebraic compute-expression reduction used by ours.
//
// Each graph firing handles one packed 8x8 block. The scan order and
// c_south/c_east line buffering mirror FPGA_PL/aie/ProcessUnit/srad_fpga.cc.

namespace {

constexpr unsigned kLaneOffsets = 0x76543210;

inline v8float load8(const float* __restrict ptr) {
    return *(reinterpret_cast<const v8float*>(ptr));
}

inline void store8(float* __restrict ptr, v8float v) {
    *(reinterpret_cast<v8float*>(ptr)) = v;
}

inline v8float splat8(float value) {
    alignas(32) float tmp[8] = {
        value, value, value, value, value, value, value, value};
    return load8(tmp);
}

inline v8float vadd(v8float acc, v8float x, v8float ones) {
    return fpmac(acc, x, 0, kLaneOffsets, ones, 0, kLaneOffsets);
}

inline v8float vmul(v8float a, v8float b) {
    return fpmul(a, 0, kLaneOffsets, b, 0, kLaneOffsets);
}

inline v8float vdiv(v8float num,
                    v8float den,
                    float* __restrict num_lane,
                    float* __restrict den_lane,
                    float* __restrict out_lane) {
    store8(num_lane, num);
    store8(den_lane, den);

    for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
            out_lane[lane] = num_lane[lane] / den_lane[lane];
        }

    return load8(out_lane);
}

inline int block_index(int r, int c) {
    return r * BLOCK_COL + c;
}

} // namespace

extern "C" void srad_coeff_update(srad_mid_input_buffer& mid_buffer,
                                  srad_scalar_input_buffer& in_q0sqr,
                                  srad_scalar_input_buffer& in_lambda,
                                  output_buffer<float>& out_j_next) {
    const float* __restrict mid_base = mid_buffer.data();
    const float* __restrict q0_base = in_q0sqr.data();
    const float* __restrict lambda_base = in_lambda.data();
    float* __restrict out_base = out_j_next.data();

    const float q0sqr = q0_base[0];
    const float lambda = lambda_base[0];

    const v8float q0 = splat8(q0sqr);
    const v8float one = splat8(srad_math::kOne);
    const v8float neg_one = splat8(-srad_math::kOne);
    const v8float quarter = splat8(srad_math::kQuarter);
    const v8float half = splat8(srad_math::kHalf);
    const v8float one_sixteenth = splat8(srad_math::kOneSixteenth);
    const v8float q0_den = vmul(q0, vadd(one, q0, one));

    alignas(32) float c_south[BLOCK_COL];
    alignas(32) float dN_lane[srad_cfg::kLanes];
    alignas(32) float dS_lane[srad_cfg::kLanes];
    alignas(32) float dW_lane[srad_cfg::kLanes];
    alignas(32) float dE_lane[srad_cfg::kLanes];
    alignas(32) float JC_lane[srad_cfg::kLanes];
    alignas(32) float c_lane[srad_cfg::kLanes];
    alignas(32) float div_num[srad_cfg::kLanes];
    alignas(32) float div_den[srad_cfg::kLanes];
    alignas(32) float div_out[srad_cfg::kLanes];
    int mid = 0;

    for (int j = 0; j < BLOCK_COL; ++j)
        chess_prepare_for_pipelining
        chess_loop_range(BLOCK_COL, BLOCK_COL) {
            c_south[j] = srad_math::kZero;
        }

    for (int i = BLOCK_ROW - 1; i >= 0; --i) {
        float c_east = srad_math::kZero;

        for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                const int j = BLOCK_COL - 1 - lane;

                ++mid;
                dN_lane[j] = mid_base[mid++];
                dS_lane[j] = mid_base[mid++];
                dW_lane[j] = mid_base[mid++];
                dE_lane[j] = mid_base[mid++];
                JC_lane[j] = mid_base[mid++];
            }

        if (srad_cfg::kBypassCoeffMath) {
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    c_lane[lane] = srad_math::kOne;
                }
        } else {
            const v8float dN_vec = load8(dN_lane);
            const v8float dS_vec = load8(dS_lane);
            const v8float dW_vec = load8(dW_lane);
            const v8float dE_vec = load8(dE_lane);
            const v8float JC_vec = load8(JC_lane);

            const v8float A_sum0 =
                vadd(vmul(dN_vec, dN_vec), vmul(dS_vec, dS_vec), one);
            const v8float A_sum1 =
                vadd(vmul(dW_vec, dW_vec), vmul(dE_vec, dE_vec), one);
            const v8float A_vec = vadd(A_sum0, A_sum1, one);
            const v8float JC2_vec = vmul(JC_vec, JC_vec);
            const v8float g2_vec =
                vdiv(A_vec, JC2_vec, div_num, div_den, div_out);

            const v8float B_sum0 = vadd(dN_vec, dS_vec, one);
            const v8float B_sum1 = vadd(dW_vec, dE_vec, one);
            const v8float B_vec = vadd(B_sum0, B_sum1, one);
            const v8float lap_vec =
                vdiv(B_vec, JC_vec, div_num, div_den, div_out);
            const v8float lap2_vec = vmul(lap_vec, lap_vec);
            const v8float num_vec =
                vadd(vmul(half, g2_vec), vmul(one_sixteenth, lap2_vec), neg_one);
            const v8float den_base =
                vadd(one, vmul(quarter, lap_vec), one);
            const v8float den2_vec = vmul(den_base, den_base);
            const v8float qsqr_vec =
                vdiv(num_vec, den2_vec, div_num, div_den, div_out);
            const v8float ratio_num = vadd(qsqr_vec, q0, neg_one);
            const v8float ratio_vec =
                vdiv(ratio_num, q0_den, div_num, div_den, div_out);
            const v8float c_den = vadd(one, ratio_vec, one);
            const v8float c_div =
                vdiv(one, c_den, div_num, div_den, div_out);

            store8(c_lane, c_div);
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    c_lane[lane] = srad_math::clamp01(c_lane[lane]);
                }
        }

        for (int j = BLOCK_COL - 1; j >= 0; --j)
            chess_prepare_for_pipelining
            chess_loop_range(BLOCK_COL, BLOCK_COL) {
                const int idx = block_index(i, j);
                const float c_here = c_lane[j];
                const float dN = dN_lane[j];
                const float dS = dS_lane[j];
                const float dW = dW_lane[j];
                const float dE = dE_lane[j];
                const float JC = JC_lane[j];
                const float cN = c_here;
                const float cW = c_here;
                const float cS = (i == BLOCK_ROW - 1) ? c_here : c_south[j];
                const float cE = (j == BLOCK_COL - 1) ? c_here : c_east;
                const float D = cN * dN + cS * dS + cW * dW + cE * dE;

                out_base[idx] = JC + srad_math::kQuarter * lambda * D;
                c_south[j] = c_here;
                c_east = c_here;
            }
    }
}
