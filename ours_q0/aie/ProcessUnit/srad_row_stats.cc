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

extern "C" void srad_row_stats(srad_stats_input_buffer& in_j,
                                srad_stats_output_buffer& out_stats) {
#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    static int debug_call = 0;
    const int call_id = debug_call++;
#endif

    const float* __restrict row = in_j.data();
    float* __restrict stats = out_stats.data();

    const v8float zero = splat(srad_math::kZero);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    aie::vector<float, srad_cfg::kLanes> row_sum_v(zero);
    aie::vector<float, srad_cfg::kLanes> row_sum2_v(zero);

    for (int chunk = 0; chunk < kChunksPerRow; ++chunk)
        chess_prepare_for_pipelining
        chess_loop_range(kChunksPerRow, kChunksPerRow) {
        const int col = chunk * srad_cfg::kLanes;
        const auto data_mask = make_data_lane_mask(col);
        const v8float value_native =
            select_data_lanes(load_vec(row, col), data_mask, zero_v);
        const aie::vector<float, srad_cfg::kLanes> values(value_native);
        row_sum_v = aie::add(row_sum_v, values);
        row_sum2_v = aie::add(
            row_sum2_v,
            aie::vector<float, srad_cfg::kLanes>(
                fpmul(value_native,
                      0,
                      kLaneOffsets,
                      value_native,
                      0,
                      kLaneOffsets)));
    }

    stats[0] = aie::reduce_add(row_sum_v);
    stats[1] = aie::reduce_add(row_sum2_v);

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_row_stats] call=%d sum=%.9g sum2=%.9g\n",
                    call_id,
                    stats[0],
                    stats[1]);
    }
#endif
}
