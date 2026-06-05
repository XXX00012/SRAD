#include <adf.h>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// OpenCL-v0 reduce_kernel stage.
// The input streams are the materialized d_sums and d_sums2 arrays produced
// by prepare_kernel. The no/mul/gridDim loop below follows the original host
// launch sequence for reduce_kernel while keeping the intermediate partials in
// AIE local memory for this fixed-size baseline.

namespace {

constexpr int kThreads = srad_cfg::kOpenClReductionThreads;

inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

inline int largest_reduction_pow2(int n) {
    int df = 1;
    for (int i = 2; i <= kThreads; i *= 2) {
        if (n >= i) {
            df = i;
        }
    }
    return df;
}

inline void reduce_full_block(float* __restrict psum,
                              float* __restrict psum2) {
    for (int i = 2; i <= kThreads; i *= 2) {
        for (int tx = i - 1; tx < kThreads; tx += i) {
            psum[tx] = psum[tx] + psum[tx - i / 2];
            psum2[tx] = psum2[tx] + psum2[tx - i / 2];
        }
    }
}

} // namespace

void srad_reduce_kernel(srad_image_input_buffer& in_sums,
                        srad_image_input_buffer& in_sums2,
                        output_buffer<float>& out_stats) {
    const float* __restrict sums_in = in_sums.data();
    const float* __restrict sums2_in = in_sums2.data();
    float* __restrict stats_out = out_stats.data();
    float sums[srad_cfg::kPixels];
    float sums2[srad_cfg::kPixels];
    float psum[kThreads];
    float psum2[kThreads];

    for (int ei = 0; ei < srad_cfg::kPixels; ++ei)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kPixels, srad_cfg::kPixels) {
            sums[ei] = sums_in[ei];
            sums2[ei] = sums2_in[ei];
        }

    int no = srad_cfg::kPixels;
    int mul = 1;
    int grid_dim = ceil_div_int(no, kThreads);

    while (grid_dim != 0) {
        const int nf = kThreads - (grid_dim * kThreads - no);

        for (int bx = 0; bx < grid_dim; ++bx) {
            for (int tx = 0; tx < kThreads; ++tx)
                chess_prepare_for_pipelining
                chess_loop_range(kThreads, kThreads) {
                    psum[tx] = 0.0f;
                    psum2[tx] = 0.0f;
                }

            for (int tx = 0; tx < kThreads; ++tx)
            chess_prepare_for_pipelining
            chess_loop_range(kThreads, kThreads) {
                const int ei = bx * kThreads + tx;
                if (ei < no) {
                    const int src = ei * mul;
                    psum[tx] = sums[src];
                    psum2[tx] = sums2[src];
                }
            }

            const int dst = bx * mul * kThreads;
            if (nf == kThreads || bx != grid_dim - 1) {
                reduce_full_block(psum, psum2);
                sums[dst] = psum[kThreads - 1];
                sums2[dst] = psum2[kThreads - 1];
            } else {
                const int df = largest_reduction_pow2(nf);
                for (int i = 2; i <= df; i *= 2) {
                    for (int tx = i - 1; tx < df; tx += i) {
                        psum[tx] = psum[tx] + psum[tx - i / 2];
                        psum2[tx] = psum2[tx] + psum2[tx - i / 2];
                    }
                }

                const int tx = df - 1;
                for (int i = bx * kThreads + df;
                     i < bx * kThreads + nf;
                     ++i) {
                    psum[tx] = psum[tx] + sums[i];
                    psum2[tx] = psum2[tx] + sums2[i];
                }
                sums[dst] = psum[tx];
                sums2[dst] = psum2[tx];
            }
        }

        no = grid_dim;
        if (grid_dim == 1) {
            grid_dim = 0;
        } else {
            mul *= kThreads;
            grid_dim = ceil_div_int(no, kThreads);
        }
    }

    stats_out[0] = sums[0];
    stats_out[1] = sums2[0];
    for (int lane = 2; lane < srad_cfg::kScalarPacketElems; ++lane) {
        stats_out[lane] = 0.0f;
    }
}
