#include <adf.h>

#include <ProcessUnit/include.h>
#include <ProcessUnit/srad.h>

void srad_update_kernel(srad_float_input_buffer& in_update,
                        srad_float_input_buffer& in_meta,
                        output_buffer<float>& out_i_next) {
    const float* __restrict u = in_update.data();
    const float lambda = in_meta.data()[0];

    const float* __restrict image = u + srad_cfg::kUpdatePlaneI;
    const float* __restrict dN = u + srad_cfg::kUpdatePlaneDn;
    const float* __restrict dS = u + srad_cfg::kUpdatePlaneDs;
    const float* __restrict dW = u + srad_cfg::kUpdatePlaneDw;
    const float* __restrict dE = u + srad_cfg::kUpdatePlaneDe;
    const float* __restrict cN = u + srad_cfg::kUpdatePlaneCn;
    const float* __restrict cS = u + srad_cfg::kUpdatePlaneCs;
    const float* __restrict cE = u + srad_cfg::kUpdatePlaneCe;

    float* __restrict next = out_i_next.data();
    const float scale = srad_math::kQuarter * lambda;

    for (int tx = 0; tx < srad_cfg::kCudaBlockElems; ++tx)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kCudaBlockElems,
                         srad_cfg::kCudaBlockElems) {
        const float c_w = cN[tx];
        const float divergence =
            (cN[tx] * dN[tx]) +
            (cS[tx] * dS[tx]) +
            (c_w * dW[tx]) +
            (cE[tx] * dE[tx]);
        next[tx] = image[tx] + (scale * divergence);
    }
}
