#include <adf.h>

#include <ProcessUnit/include.h>
#include <ProcessUnit/srad.h>

namespace {

int clamp_valid_count(float raw_count) {
    int valid = static_cast<int>(raw_count);
    if (valid < 0) {
        valid = 0;
    }
    if (valid > srad_cfg::kCudaBlockElems) {
        valid = srad_cfg::kCudaBlockElems;
    }
    return valid;
}

}  // namespace

void srad_reduce_kernel(srad_reduce_input_buffer& in_packet,
                        srad_reduce_output_buffer& out_partial) {
    const float* __restrict packet = in_packet.data();
    const float* __restrict pairs = packet + srad_cfg::kMetaPacketElems;
    float* __restrict partial = out_partial.data();

    const int valid_count = clamp_valid_count(packet[0]);
    float sum = srad_math::kZero;
    float sum2 = srad_math::kZero;

    for (int tx = 0; tx < srad_cfg::kCudaBlockElems; ++tx)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kCudaBlockElems,
                         srad_cfg::kCudaBlockElems) {
        if (tx < valid_count) {
            sum += pairs[(2 * tx) + 0];
            sum2 += pairs[(2 * tx) + 1];
        }
    }

    partial[0] = sum;
    partial[1] = sum2;
}
