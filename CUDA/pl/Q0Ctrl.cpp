#include "../aie/Config.h"

// Legacy placeholder: the active CUDA-style SRAD board path computes q0 in PS
// from ReducePL partials. This kernel is intentionally not built by Makefile.

extern "C" {

void Q0Ctrl(float* debug, int iter_cnt) {
#pragma HLS INTERFACE m_axi port=debug offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=debug bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (debug != nullptr) {
        debug[0] = static_cast<float>(iter_cnt);
        debug[1] = static_cast<float>(srad_cfg::kCudaBlockElems);
    }
}

}
