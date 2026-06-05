#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// FPGA-v5-style single AIE fused baseline using CUDA float32 math.
// Phase 1 computes q0sqr inside AIE; phase 2 fuses SRAD1/SRAD2.

namespace {

inline float compute_q0sqr(const float* __restrict j_reduce) {
    float sum = 0.0f;
    float sum2 = 0.0f;

    constexpr int kRedUnroll = 8;
    constexpr int kRedExit =
        (srad_cfg::kPixels + kRedUnroll - 1) / kRedUnroll;

    for (int i = 0; i < kRedExit; ++i) {
        float group_sum = 0.0f;
        float group_sum2 = 0.0f;

        for (int j = 0; j < kRedUnroll; ++j)
            chess_prepare_for_pipelining
            chess_loop_range(kRedUnroll, kRedUnroll) {
                const int idx = i * kRedUnroll + j;
                const float v = (idx < srad_cfg::kPixels) ? j_reduce[idx] : 0.0f;
                group_sum += v;
                group_sum2 += v * v;
            }

        sum += group_sum;
        sum2 += group_sum2;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    return variance / (mean * mean);
}

inline float compute_c(float jc,
                       float dN,
                       float dS,
                       float dW,
                       float dE,
                       float q0sqr) {
    const float g2 =
        (dN * dN + dS * dS + dW * dW + dE * dE) / (jc * jc);
    const float l = (dN + dS + dW + dE) / jc;
    const float num = 0.5f * g2 - (1.0f / 16.0f) * l * l;
    float den = 1.0f + 0.25f * l;
    const float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    const float c = 1.0f / (1.0f + den);

    return srad_math::clamp01(c);
}

} // namespace

void srad_fpga_v5_fused(srad_image_input_buffer& in_reduce,
                        srad_image_input_buffer& in_compute,
                        srad_scalar_input_buffer& in_lambda,
                        srad_image_output_buffer& out_j_next) {
    const float* __restrict reduce_base = in_reduce.data();
    const float* __restrict compute_base = in_compute.data();
    const float* __restrict lambda_base = in_lambda.data();
    float* __restrict out_base = out_j_next.data();

    const float lambda = lambda_base[0];
    const float q0sqr = compute_q0sqr(reduce_base);

    alignas(32) float c_south[COL];

    for (int j = 0; j < COL; ++j)
        chess_prepare_for_pipelining
        chess_loop_range(COL, COL) {
            c_south[j] = 0.0f;
        }

    for (int i = ROW - 1; i >= 0; --i) {
        const int iN = srad_math::north_row(i);
        const int iS = srad_math::south_row(i);
        float c_east = 0.0f;

        for (int j = COL - 1; j >= 0; --j)
            chess_prepare_for_pipelining
            chess_loop_range(COL, COL) {
                const int jW = (j == 0) ? 0 : j - 1;
                const int jE = (j == COL - 1) ? COL - 1 : j + 1;
                const int idx = srad_math::image_index(i, j);

                const float JC = compute_base[idx];
                const float JN = compute_base[srad_math::image_index(iN, j)];
                const float JS = compute_base[srad_math::image_index(iS, j)];
                const float JW = compute_base[srad_math::image_index(i, jW)];
                const float JE = compute_base[srad_math::image_index(i, jE)];

                const float dN = JN - JC;
                const float dS = JS - JC;
                const float dW = JW - JC;
                const float dE = JE - JC;
                const float c_here = srad_cfg::kBypassCoeffMath
                                         ? 1.0f
                                         : compute_c(JC, dN, dS, dW, dE, q0sqr);

                const float cN = c_here;
                const float cW = c_here;
                const float cS = (i == ROW - 1) ? c_here : c_south[j];
                const float cE = (j == COL - 1) ? c_here : c_east;

                const float D = cN * dN + cS * dS + cW * dW + cE * dE;
                out_base[idx] = JC + 0.25f * lambda * D;

                c_south[j] = c_here;
                c_east = c_here;
            }
    }
}
