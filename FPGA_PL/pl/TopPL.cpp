#include "../aie/Config.h"

#include <hls_stream.h>

extern "C" {

void LoadFpgaV5PLQ0(const float* image,
                    float lambda,
                    hls::stream<float>& out_compute,
                    hls::stream<float>& out_q0sqr,
                    hls::stream<float>& out_lambda) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_compute
#pragma HLS INTERFACE axis port=out_q0sqr
#pragma HLS INTERFACE axis port=out_lambda
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=lambda bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    constexpr int kRedUnroll = 8;
    constexpr int kRedExit =
        (srad_cfg::kPixels + kRedUnroll - 1) / kRedUnroll;

    float sum = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < kRedExit; ++i) {
        float group_sum = 0.0f;
        float group_sum2 = 0.0f;

        for (int j = 0; j < kRedUnroll; ++j) {
#pragma HLS PIPELINE II=1
            const int idx = i * kRedUnroll + j;
            const float v = (idx < srad_cfg::kPixels) ? image[idx] : 0.0f;
            if (idx < srad_cfg::kPixels) {
                out_compute.write(v);
            }
            group_sum += v;
            group_sum2 += v * v;
        }

        sum += group_sum;
        sum2 += group_sum2;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    const float q0sqr = variance / (mean * mean);

    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        out_q0sqr.write((i == 0) ? q0sqr : 0.0f);
        out_lambda.write((i == 0) ? lambda : 0.0f);
    }
}

void StoreFpgaV5PLQ0(hls::stream<float>& in_j_next,
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
