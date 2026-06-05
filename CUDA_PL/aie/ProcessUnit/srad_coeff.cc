#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// OpenCL-v0 faithful float32 SRAD coefficient stage.
// This kernel corresponds to srad_kernel and materializes:
//   d_c, d_dN, d_dS, d_dW, d_dE
// as five independent PLIO-backed OpenCL-layout arrays. Each invocation
// corresponds to one 256-work-item OpenCL block and receives the full d_I
// image, so neighbor addressing stays inside this kernel.

namespace {

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

void srad_kernel(srad_image_input_buffer& in_j,
                 srad_scalar_input_buffer& in_q0sqr,
                 output_buffer<float>& out_c,
                 output_buffer<float>& out_dN,
                 output_buffer<float>& out_dS,
                 output_buffer<float>& out_dW,
                 output_buffer<float>& out_dE) {
    const float* __restrict j_base = in_j.data();
    const float* __restrict q0_base = in_q0sqr.data();
    float* __restrict c_base = out_c.data();
    float* __restrict dN_base = out_dN.data();
    float* __restrict dS_base = out_dS.data();
    float* __restrict dW_base = out_dW.data();
    float* __restrict dE_base = out_dE.data();

    const float q0sqr = q0_base[0];
    static int block_id = 0;
    const int bx = block_id;
    block_id = (block_id + 1) % srad_cfg::kTileCount;

    for (unsigned idx = 0; idx < srad_cfg::kBlockPixels; ++idx)
        chess_prepare_for_pipelining
        chess_loop_range(1, ) {
            const int ei = bx * srad_cfg::kBlockPixels + idx;
            const int i = ei % srad_cfg::kRows;
            const int j = ei / srad_cfg::kRows;
            const int iN = srad_math::north_row(i);
            const int iS = srad_math::south_row(i);
            const int jW = srad_math::west_col(j);
            const int jE = srad_math::east_col(j);

            const float JC = j_base[ei];
            const float JN = j_base[srad_math::image_index(iN, j)];
            const float JS = j_base[srad_math::image_index(iS, j)];
            const float JW = j_base[srad_math::image_index(i, jW)];
            const float JE = j_base[srad_math::image_index(i, jE)];

            const float dN = JN - JC;
            const float dS = JS - JC;
            const float dW = JW - JC;
            const float dE = JE - JC;

            dN_base[idx] = dN;
            dS_base[idx] = dS;
            dW_base[idx] = dW;
            dE_base[idx] = dE;
            c_base[idx] = srad_cfg::kBypassCoeffMath
                               ? 1.0f
                               : compute_c(JC, dN, dS, dW, dE, q0sqr);
        }
}
