#include <adf.h>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// OpenCL-v0 prepare_kernel:
//   d_sums[ei] = d_I[ei]
//   d_sums2[ei] = d_I[ei] * d_I[ei]

void srad_prepare_kernel(srad_image_input_buffer& in_j,
                         output_buffer<float>& out_sums,
                         output_buffer<float>& out_sums2) {
    const float* __restrict j_base = in_j.data();
    float* __restrict sums_base = out_sums.data();
    float* __restrict sums2_base = out_sums2.data();

    for (int ei = 0; ei < srad_cfg::kPixels; ++ei)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kPixels, srad_cfg::kPixels) {
            const float v = j_base[ei];
            sums_base[ei] = v;
            sums2_base[ei] = v * v;
        }
}
