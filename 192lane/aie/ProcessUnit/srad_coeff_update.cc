#include <adf.h>
#include <cstdint>
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

static_assert((kChunksPerRow & (kChunksPerRow - 1)) == 0,
              "coeff-update next chunk wrap expects a power-of-two chunk count");

inline v8float decode_coeff_vec(
    v8float value_or_num,
    v8float tag_or_den,
    aie::vector<float, srad_cfg::kLanes> zero_v,
    aie::vector<float, srad_cfg::kLanes> one_v) {
    const aie::vector<float, srad_cfg::kLanes> value_v(value_or_num);
    const aie::vector<float, srad_cfg::kLanes> tag_v(tag_or_den);
    const auto raw_mask = aie::lt(zero_v, tag_v);
    const auto safe_den = aie::select(one_v, tag_v, raw_mask);
    const auto divided =
        aie::mul(value_v, aie::inv(safe_den)).template to_vector<float>();
    return aie::select(value_v, divided, raw_mask).to_native();
}

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

inline uint32_t float_to_bits(float value) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = value;
    return conv.u;
}

} // namespace

extern "C" void srad_coeff_update(srad_mid_input_buffer& in_c,
                                  srad_update_input_buffer& in_j,
                                  output_pktstream* out_j_next) {
#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    static int debug_call = 0;
    const int call_id = debug_call++;
    if (call_id < 8) {
        std::printf("[srad_coeff_update] enter call=%d\n", call_id);
    }
#endif

    const float* __restrict mid = in_c.data();
    const float* __restrict center_value =
        mid + kCenterValuePlane * srad_cfg::kMidPlaneStride;
    const float* __restrict center_tag =
        mid + kCenterTagPlane * srad_cfg::kMidPlaneStride;
    const float* __restrict south_value =
        mid + kSouthValuePlane * srad_cfg::kMidPlaneStride;
    const float* __restrict south_tag =
        mid + kSouthTagPlane * srad_cfg::kMidPlaneStride;

    const float* __restrict base = in_j.data();
    const float* __restrict row_m_minus1 = base + 0 * srad_cfg::kRowPhysElems;
    const float* __restrict row_m = base + 1 * srad_cfg::kRowPhysElems;
    const float* __restrict row_m_plus1 = base + 2 * srad_cfg::kRowPhysElems;

    alignas(32) float out_row[srad_cfg::kOutputRowPhysElems];
    float* __restrict out = out_row;
    float row_sum = 0.0f;
    float row_sum2 = 0.0f;

    const v8float zero = splat(srad_math::kZero);
    const v8float one = splat(srad_math::kOne);
    const v8float neg_one = splat(-srad_math::kOne);
    const v8float update_scale =
        splat(srad_math::kQuarter * srad_cfg::kLambdaDefault);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);

    v8float prev_row_m = zero;
    for (int chunk = 0; chunk < kChunksPerRow; ++chunk)
        chess_prepare_for_pipelining
        chess_loop_range(kChunksPerRow, kChunksPerRow) {
        const int col = chunk * srad_cfg::kLanes;
        const int next_col =
            ((chunk + 1) & (kChunksPerRow - 1)) * srad_cfg::kLanes;
        const auto data_mask = make_data_lane_mask(col);
        const auto next_data_mask = make_data_lane_mask(next_col);
        const v8float raw_jc = load_vec(row_m, col);
        const v8float raw_jn = load_vec(row_m_minus1, col);
        const v8float raw_js = load_vec(row_m_plus1, col);
        const v8float raw_next_jc = load_vec(row_m, next_col);

        const v8float jc = select_data_lanes(raw_jc, data_mask, zero_v);
        const v8float jn = select_data_lanes(raw_jn, data_mask, zero_v);
        const v8float js = select_data_lanes(raw_js, data_mask, zero_v);
        const v8float next_row_m =
            select_data_lanes(raw_next_jc, next_data_mask, zero_v);
        const v8float jw = select_data_lanes(
            shift_left_with_zero(jc, prev_row_m, one),
            data_mask,
            zero_v);
        const v8float je = select_data_lanes(
            shift_right_with_zero(jc, next_row_m, one),
            data_mask,
            zero_v);

        const v8float d_n =
            fpmac(jn, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float d_s =
            fpmac(js, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float d_w =
            fpmac(jw, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

        const v8float coeff = decode_coeff_vec(
            load_vec(center_value, col),
            load_vec(center_tag, col),
            zero_v, one_v);
        const v8float coeff_south = decode_coeff_vec(
            load_vec(south_value, col),
            load_vec(south_tag, col),
            zero_v, one_v);
        const v8float coeff_next = decode_coeff_vec(
            load_vec(center_value, next_col),
            load_vec(center_tag, next_col),
            zero_v, one_v);
        const v8float coeff_east = select_data_lanes(
            shift_right_with_zero(coeff, coeff_next, one),
            data_mask,
            zero_v);
        const v8float d_e =
            fpmac(je, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

        const v8float da =
            fpmac(fpmul(coeff, 0, kLaneOffsets, d_n, 0, kLaneOffsets),
                  fpmul(coeff_south, 0, kLaneOffsets, d_s, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float db =
            fpmac(fpmul(coeff, 0, kLaneOffsets, d_w, 0, kLaneOffsets),
                  fpmul(coeff_east, 0, kLaneOffsets, d_e, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float divergence =
            fpmac(da, db, 0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float next =
            fpmac(jc,
                  fpmul(update_scale, 0, kLaneOffsets, divergence, 0,
                        kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);

        const v8float next_masked = select_data_lanes(next, data_mask, zero_v);
        store_vec(out, col, next_masked);
        accumulate_stats(next_masked, row_sum, row_sum2);
        prev_row_m = jc;
    }

    out[srad_cfg::kStatSumPadIndex] = row_sum;
    out[srad_cfg::kStatSum2PadIndex] = row_sum2;

    const uint32_t packet_id = getPacketid(out_j_next, 0);
    writeHeader(out_j_next, 0, packet_id);
    for (int i = 0; i < srad_cfg::kOutputRowPhysElems; ++i)
        chess_prepare_for_pipelining
        writeincr(out_j_next,
                  float_to_bits(out_row[i]),
                  i == (srad_cfg::kOutputRowPhysElems - 1));

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_coeff_update] exit call=%d out0=%.9g sum=%.9g sum2=%.9g\n",
                    call_id,
                    out_row[0],
                    row_sum,
                    row_sum2);
    }
#endif
}
