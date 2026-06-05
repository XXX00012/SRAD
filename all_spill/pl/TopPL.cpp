#include "../aie/Config.h"

#include <hls_stream.h>

// Ours-undivide PL shim:
//   DDR image -> PL q0sqr -> PLIO packets for AIE -> PLIO result -> DDR.

extern "C" {

void TopPL(const float* image,
           float* output,
           float lambda,
           hls::stream<float>& out_j,
           hls::stream<float>& out_j_update,
           hls::stream<float>& out_q0sqr,
           hls::stream<float>& out_lambda,
           hls::stream<float>& in_j_next) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=out_j_update
#pragma HLS INTERFACE axis port=out_q0sqr
#pragma HLS INTERFACE axis port=out_lambda
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=lambda bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float sum = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = image[i];
        sum += v;
        sum2 += v * v;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    const float q0sqr = (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;

    for (int blk = 0; blk < srad_cfg::kBlockCount; ++blk) {
        for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
            out_q0sqr.write((i == 0) ? q0sqr : 0.0f);
            out_lambda.write((i == 0) ? lambda : 0.0f);
        }
    }

    for (int br = 0; br < srad_cfg::kRows; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < srad_cfg::kCols; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
#pragma HLS PIPELINE II=1
                    const int idx = (br + r) * srad_cfg::kCols + (bc + c);
                    const float v = image[idx];
                    out_j.write(v);
                    out_j_update.write(v);
                }
            }
        }
    }

    for (int br = 0; br < srad_cfg::kRows; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < srad_cfg::kCols; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
#pragma HLS PIPELINE II=1
                    const int idx = (br + r) * srad_cfg::kCols + (bc + c);
                    output[idx] = in_j_next.read();
                }
            }
        }
    }
}

}
