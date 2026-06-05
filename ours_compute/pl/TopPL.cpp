#include "../aie/Config.h"

// Ours Phase 1 in PL: srad_q0_pl computes global float32 q0sqr.
// The q0_debug M_AXI output is both host-validation data and the formal q0
// source for AIE: host copies this PL-computed scalar packet to AIE through a
// normal GMIO input.

extern "C" {

void TopPL(const float* j_reduce, float* q0_debug) {
#pragma HLS INTERFACE m_axi port=j_reduce offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=q0_debug offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=j_reduce bundle=control
#pragma HLS INTERFACE s_axilite port=q0_debug bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float sum = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = j_reduce[i];
        sum += v;
        sum2 += v * v;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    const float q0sqr = (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;

    q0_debug[0] = q0sqr;
    for (int i = 1; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        q0_debug[i] = 0.0f;
    }
}

}
