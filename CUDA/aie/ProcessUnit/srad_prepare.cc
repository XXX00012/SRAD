#include <adf.h>

#include <ProcessUnit/include.h>
#include <ProcessUnit/srad.h>

void srad_prepare_kernel(srad_float_input_buffer& in_i,
                         output_buffer<float>& out_sums,
                         output_buffer<float>& out_sums2) {
    const float* __restrict image = in_i.data();
    float* __restrict sums = out_sums.data();
    float* __restrict sums2 = out_sums2.data();

    for (int tx = 0; tx < srad_cfg::kCudaBlockElems; ++tx)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kCudaBlockElems,
                         srad_cfg::kCudaBlockElems) {
        const float value = image[tx];
        sums[tx] = value;
        sums2[tx] = value * value;
    }
}
