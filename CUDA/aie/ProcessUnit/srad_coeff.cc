#include <adf.h>

#include <ProcessUnit/include.h>
#include <ProcessUnit/srad.h>

namespace {

float clamp_coeff(float value) {
    if (value < srad_math::kZero) {
        return srad_math::kZero;
    }
    if (value > srad_math::kOne) {
        return srad_math::kOne;
    }
    return value;
}

}  // namespace

void srad_coeff_kernel(srad_float_input_buffer& in_neighbors,
                       srad_float_input_buffer& in_q0,
                       output_buffer<float>& out_dN,
                       output_buffer<float>& out_dS,
                       output_buffer<float>& out_dW,
                       output_buffer<float>& out_dE,
                       output_buffer<float>& out_c) {
    const float* __restrict n = in_neighbors.data();
    const float q0sqr = in_q0.data()[0];

    const float* __restrict jc = n + srad_cfg::kCoeffPlaneJc;
    const float* __restrict jn = n + srad_cfg::kCoeffPlaneJn;
    const float* __restrict js = n + srad_cfg::kCoeffPlaneJs;
    const float* __restrict jw = n + srad_cfg::kCoeffPlaneJw;
    const float* __restrict je = n + srad_cfg::kCoeffPlaneJe;

    float* __restrict dN = out_dN.data();
    float* __restrict dS = out_dS.data();
    float* __restrict dW = out_dW.data();
    float* __restrict dE = out_dE.data();
    float* __restrict c = out_c.data();

    for (int tx = 0; tx < srad_cfg::kCudaBlockElems; ++tx)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kCudaBlockElems,
                         srad_cfg::kCudaBlockElems) {
        const float j_c = jc[tx];
        const float d_n = jn[tx] - j_c;
        const float d_s = js[tx] - j_c;
        const float d_w = jw[tx] - j_c;
        const float d_e = je[tx] - j_c;

        const float g2 =
            (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e) /
            (j_c * j_c);
        const float l = (d_n + d_s + d_w + d_e) / j_c;
        const float num =
            (srad_math::kHalf * g2) -
            (srad_math::kOneSixteenth * (l * l));
        const float den0 = srad_math::kOne + (srad_math::kQuarter * l);
        const float qsqr = num / (den0 * den0);
        const float den1 =
            (qsqr - q0sqr) / (q0sqr * (srad_math::kOne + q0sqr));

        dN[tx] = d_n;
        dS[tx] = d_s;
        dW[tx] = d_w;
        dE[tx] = d_e;
        c[tx] = clamp_coeff(srad_math::kOne / (srad_math::kOne + den1));
    }
}
