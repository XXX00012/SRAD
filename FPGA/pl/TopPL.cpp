#include "../aie/Config.h"

#include <hls_stream.h>

extern "C" {

void LoadFpgaV5(const float* image,
                float lambda,
                hls::stream<float>& out_reduce,
                hls::stream<float>& out_compute,
                hls::stream<float>& out_lambda) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_reduce
#pragma HLS INTERFACE axis port=out_compute
#pragma HLS INTERFACE axis port=out_lambda
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=lambda bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = image[i];
        out_reduce.write(v);
        out_compute.write(v);
    }

    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        out_lambda.write((i == 0) ? lambda : 0.0f);
    }
}

void StoreFpgaV5(hls::stream<float>& in_j_next,
                 float* output) {
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = in_j_next.read();
    }
}

}
