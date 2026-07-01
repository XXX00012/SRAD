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

static_assert((kChunksPerRow & (kChunksPerRow - 1)) == 0,
              "coeff-update next chunk wrap expects a power-of-two chunk count");

inline v8float decode_coeff_vec(
    v8float value_or_q2,
    v8float tag_or_valid,
    v8float q0sqr,
    v8float neg_one,
    aie::vector<float, srad_cfg::kLanes> q0_den_v,
    aie::vector<float, srad_cfg::kLanes> zero_v,
    aie::vector<float, srad_cfg::kLanes> one_v) {
    const aie::vector<float, srad_cfg::kLanes> value_v(value_or_q2);
    const aie::vector<float, srad_cfg::kLanes> tag_v(tag_or_valid);
    const auto raw_mask = aie::lt(zero_v, tag_v);
    const auto q0_pos_mask = aie::lt(zero_v, q0_den_v);

    const auto safe_q0_den = aie::select(one_v, q0_den_v, q0_pos_mask);
    const v8float q0_delta =
        fpmac(value_or_q2, q0sqr, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const aie::vector<float, srad_cfg::kLanes> q0_delta_v(q0_delta);
    const auto q0_ratio =
        aie::mul(q0_delta_v, aie::inv(safe_q0_den)).template to_vector<float>();
    const v8float q0_ratio_native = q0_ratio.to_native();
    const v8float one = one_v.to_native();
    const v8float coeff_den_native =
        fpmac(one, q0_ratio_native, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const aie::vector<float, srad_cfg::kLanes> coeff_den(coeff_den_native);
    const auto safe_coeff_den = aie::select(one_v, coeff_den, q0_pos_mask);
    const auto raw_coeff =
        aie::mul(one_v, aie::inv(safe_coeff_den)).template to_vector<float>();

    const auto above_one_mask = aie::lt(one_v, raw_coeff);
    const auto below_zero_mask = aie::lt(raw_coeff, zero_v);
    auto clamped = aie::select(raw_coeff, one_v, above_one_mask);
    clamped = aie::select(clamped, zero_v, below_zero_mask);

    const auto q0_safe_coeff = aie::select(one_v, clamped, q0_pos_mask);
    return aie::select(value_v, q0_safe_coeff, raw_mask).to_native();
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

} // namespace

extern "C" void srad_coeff_update(srad_mid_input_buffer& in_c,
                                  srad_update_input_buffer& in_j,
                                  output_buffer<float>& out_j_next) {
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

    const float q0sqr = row_m[srad_cfg::kQ0PadIndex];
    const float q0_den_scalar = q0sqr * (srad_math::kOne + q0sqr);

    float* __restrict out = out_j_next.data();
    float row_sum = 0.0f;
    float row_sum2 = 0.0f;

    const v8float zero = splat(srad_math::kZero);
    const v8float one = splat(srad_math::kOne);
    const v8float neg_one = splat(-srad_math::kOne);
    const v8float q0sqr_vec = splat(q0sqr);
    const v8float q0_den = splat(q0_den_scalar);
    const v8float update_scale =
        splat(srad_math::kQuarter * srad_cfg::kLambdaDefault);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);
    const aie::vector<float, srad_cfg::kLanes> q0_den_v(q0_den);

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
            q0sqr_vec, neg_one, q0_den_v,
            zero_v, one_v);
        const v8float coeff_south = decode_coeff_vec(
            load_vec(south_value, col),
            load_vec(south_tag, col),
            q0sqr_vec, neg_one, q0_den_v,
            zero_v, one_v);
        const v8float coeff_next = decode_coeff_vec(
            load_vec(center_value, next_col),
            load_vec(center_tag, next_col),
            q0sqr_vec, neg_one, q0_den_v,
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

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_coeff_update] exit call=%d out0=%.9g sum=%.9g sum2=%.9g\n",
                    call_id,
                    out[0],
                    row_sum,
                    row_sum2);
    }
#endif
}
