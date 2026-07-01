#include <adf.h>
#ifndef SRAD_AIE_DEBUG
#define SRAD_AIE_DEBUG 0
#endif

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
#include <cstdio>
#endif

#define SRAD_ENABLE_AIE_VEC_HELPERS 1
#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;
using namespace srad_vec;

namespace {

inline void accumulate_stats(v8float value, float& sum, float& sum2) {
    alignas(32) float lane[srad_cfg::kLanes];
    *(reinterpret_cast<v8float*>(lane)) = value;

    for (int i = 0; i < srad_cfg::kLanes; ++i)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
        sum += lane[i];
        sum2 += lane[i] * lane[i];
    }
}

} // namespace

extern "C" void srad_row_stats(srad_row_input_buffer& in_j_next,
                               output_buffer<float>& out_j_next_stats) {
#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    static int debug_call = 0;
    const int call_id = debug_call++;
    if (call_id < 8) {
        std::printf("[srad_row_stats] enter call=%d\n", call_id);
    }
#endif

    const float* __restrict in = in_j_next.data();
    float* __restrict out = out_j_next_stats.data();

    float row_sum = 0.0f;
    float row_sum2 = 0.0f;

    const v8float zero = splat(srad_math::kZero);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);

    for (int chunk = 0; chunk < kChunksPerRow; ++chunk)
        chess_prepare_for_pipelining
        chess_loop_range(kChunksPerRow, kChunksPerRow) {
        const int col = chunk * srad_cfg::kLanes;
        const auto data_mask = make_data_lane_mask(col);
        const v8float next_masked =
            select_data_lanes(load_vec(in, col), data_mask, zero_v);
        store_vec(out, col, next_masked);
        accumulate_stats(next_masked, row_sum, row_sum2);
    }

    out[srad_cfg::kStatSumPadIndex] = row_sum;
    out[srad_cfg::kStatSum2PadIndex] = row_sum2;
    out[srad_cfg::kRowPhysElems - 1] = 0.0f;

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_row_stats] exit call=%d out0=%.9g sum=%.9g sum2=%.9g\n",
                    call_id,
                    out[0],
                    row_sum,
                    row_sum2);
    }
#endif
}
