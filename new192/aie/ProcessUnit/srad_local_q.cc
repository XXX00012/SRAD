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

inline void encode_coeff_vec(v8float j_c,
                             v8float j_n,
                             v8float j_s,
                             v8float j_w,
                             v8float j_e,
                             v8float q0sqr2,
                             v8float q0_den,
                             v8float one,
                             v8float neg_one,
                             v8float quarter,
                             v8float half,
                             v8float one_sixteenth,
                             aie::vector<float, srad_cfg::kLanes> zero_v,
                             aie::vector<float, srad_cfg::kLanes> one_v,
                             aie::mask<srad_cfg::kLanes> q0_zero_mask,
                             aie::mask<srad_cfg::kLanes> bypass_mask,
                             v8float* value_out,
                             v8float* tag_out) {
    const v8float d_n =
        fpmac(j_n, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_s =
        fpmac(j_s, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_w =
        fpmac(j_w, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_e =
        fpmac(j_e, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

    const v8float a0 =
        fpmac(fpmul(d_n, 0, kLaneOffsets, d_n, 0, kLaneOffsets),
              fpmul(d_s, 0, kLaneOffsets, d_s, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float a1 =
        fpmac(fpmul(d_w, 0, kLaneOffsets, d_w, 0, kLaneOffsets),
              fpmul(d_e, 0, kLaneOffsets, d_e, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float a =
        fpmac(a0, a1, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float b0 =
        fpmac(d_n, d_s, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float b1 =
        fpmac(d_w, d_e, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float b =
        fpmac(b0, b1, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float n =
        fpmac(fpmul(half, 0, kLaneOffsets, a, 0, kLaneOffsets),
              fpmul(one_sixteenth, 0, kLaneOffsets,
                    fpmul(b, 0, kLaneOffsets, b, 0, kLaneOffsets),
                    0, kLaneOffsets),
              0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float t =
        fpmac(j_c, fpmul(quarter, 0, kLaneOffsets, b, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float d =
        fpmul(t, 0, kLaneOffsets, t, 0, kLaneOffsets);
    const v8float den =
        fpmac(n, fpmul(q0sqr2, 0, kLaneOffsets, d, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float num =
        fpmul(q0_den, 0, kLaneOffsets, d, 0, kLaneOffsets);

    const aie::vector<float, srad_cfg::kLanes> jc_v(j_c);
    const aie::vector<float, srad_cfg::kLanes> num_v(num);
    const aie::vector<float, srad_cfg::kLanes> den_v(den);

    const auto zero_mask = aie::le(den_v, zero_v);
    const auto num_zero_mask = aie::le(num_v, zero_v);
    const auto one_mask = aie::le(den_v, num_v);
    const auto center_zero_mask = aie::le(jc_v, zero_v);

    auto value_v = aie::select(num_v, one_v, one_mask);
    value_v = aie::select(value_v, zero_v, zero_mask);
    value_v = aie::select(value_v, zero_v, num_zero_mask);
    value_v = aie::select(value_v, one_v, center_zero_mask);
    value_v = aie::select(value_v, one_v, q0_zero_mask);
    value_v = aie::select(value_v, one_v, bypass_mask);

    auto tag_v = aie::select(den_v, zero_v, one_mask);
    tag_v = aie::select(tag_v, zero_v, zero_mask);
    tag_v = aie::select(tag_v, zero_v, num_zero_mask);
    tag_v = aie::select(tag_v, zero_v, center_zero_mask);
    tag_v = aie::select(tag_v, zero_v, q0_zero_mask);
    tag_v = aie::select(tag_v, zero_v, bypass_mask);

    *value_out = value_v.to_native();
    *tag_out = tag_v.to_native();
}

} // namespace

extern "C" void srad_local_q(srad_local_q_input_buffer& in_j,
                             output_buffer<float>& out_c) {
#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    static int debug_call = 0;
    const int call_id = debug_call++;
    if (call_id < 8) {
        std::printf("[srad_local_q] enter call=%d\n", call_id);
    }
#endif

    const float* __restrict base = in_j.data();

    const float* __restrict row_m_minus1 = base + 0 * srad_cfg::kRowPhysElems;
    const float* __restrict row_m = base + 1 * srad_cfg::kRowPhysElems;
    const float* __restrict row_m_plus1 = base + 2 * srad_cfg::kRowPhysElems;
    const float* __restrict row_m_plus2 = base + 3 * srad_cfg::kRowPhysElems;

    float* __restrict mid = out_c.data();
    float* __restrict center_value =
        mid + kCenterValuePlane * srad_cfg::kMidPlaneStride;
    float* __restrict center_tag =
        mid + kCenterTagPlane * srad_cfg::kMidPlaneStride;
    float* __restrict south_value =
        mid + kSouthValuePlane * srad_cfg::kMidPlaneStride;
    float* __restrict south_tag =
        mid + kSouthTagPlane * srad_cfg::kMidPlaneStride;

    const float q0sqr = row_m[srad_cfg::kQ0PadIndex];
    const float q0sqr2_scalar = q0sqr * q0sqr;
    const float q0_den_scalar = q0sqr * (srad_math::kOne + q0sqr);

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_local_q] q0 call=%d q0sqr=%.9g\n",
                    call_id,
                    q0sqr);
    }
#endif

    const v8float zero = splat(srad_math::kZero);
    const v8float one = splat(srad_math::kOne);
    const v8float neg_one = splat(-srad_math::kOne);
    const v8float quarter = splat(srad_math::kQuarter);
    const v8float half = splat(srad_math::kHalf);
    const v8float one_sixteenth = splat(srad_math::kOneSixteenth);
    const v8float q0sqr2 = splat(q0sqr2_scalar);
    const v8float q0_den = splat(q0_den_scalar);
    const v8float bypass = splat(srad_cfg::kBypassCoeffMath
                                     ? srad_math::kOne
                                     : srad_math::kZero);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);
    const aie::vector<float, srad_cfg::kLanes> q0_den_v(q0_den);
    const aie::vector<float, srad_cfg::kLanes> bypass_v(bypass);
    const auto q0_zero_mask = aie::eq(q0_den_v, zero_v);
    const auto bypass_mask = aie::eq(bypass_v, one_v);

    v8float prev_center_c = zero;
    v8float prev_south_c = zero;

    for (int chunk = 0; chunk < kChunksPerRow; ++chunk)
        chess_prepare_for_pipelining
        chess_loop_range(kChunksPerRow, kChunksPerRow) {
        const int col = chunk * srad_cfg::kLanes;
        const int next_col =
            ((chunk + 1) & (kChunksPerRow - 1)) * srad_cfg::kLanes;

        const auto data_mask = make_data_lane_mask(col);
        const auto next_data_mask = make_data_lane_mask(next_col);

        const v8float center_n = select_data_lanes(
            load_vec(row_m_minus1, col), data_mask, zero_v);
        const v8float center_c = select_data_lanes(
            load_vec(row_m, col), data_mask, zero_v);
        const v8float center_s = select_data_lanes(
            load_vec(row_m_plus1, col), data_mask, zero_v);
        const v8float center_next = select_data_lanes(
            load_vec(row_m, next_col), next_data_mask, zero_v);

        const v8float center_w = select_data_lanes(
            shift_left_with_zero(center_c, prev_center_c, one),
            data_mask,
            zero_v);
        const v8float center_e = select_data_lanes(
            shift_right_with_zero(center_c, center_next, one),
            data_mask,
            zero_v);

        const v8float south_c = center_s;
        const v8float south_n = center_c;
        const v8float south_s = select_data_lanes(
            load_vec(row_m_plus2, col), data_mask, zero_v);
        const v8float south_next = select_data_lanes(
            load_vec(row_m_plus1, next_col), next_data_mask, zero_v);
        const v8float south_w = select_data_lanes(
            shift_left_with_zero(south_c, prev_south_c, one),
            data_mask,
            zero_v);
        const v8float south_e = select_data_lanes(
            shift_right_with_zero(south_c, south_next, one),
            data_mask,
            zero_v);

        v8float center_value_vec;
        v8float center_tag_vec;
        encode_coeff_vec(center_c, center_n, center_s, center_w, center_e,
                         q0sqr2, q0_den, one, neg_one, quarter, half,
                         one_sixteenth, zero_v, one_v, q0_zero_mask,
                         bypass_mask, &center_value_vec, &center_tag_vec);

        v8float south_value_vec;
        v8float south_tag_vec;
        encode_coeff_vec(south_c, south_n, south_s, south_w, south_e,
                         q0sqr2, q0_den, one, neg_one, quarter, half,
                         one_sixteenth, zero_v, one_v, q0_zero_mask,
                         bypass_mask, &south_value_vec, &south_tag_vec);

        store_vec(center_value, col,
                  select_data_lanes(center_value_vec, data_mask, one_v));
        store_vec(center_tag, col,
                  select_data_lanes(center_tag_vec, data_mask, zero_v));
        store_vec(south_value, col,
                  select_data_lanes(south_value_vec, data_mask, zero_v));
        store_vec(south_tag, col,
                  select_data_lanes(south_tag_vec, data_mask, zero_v));

        prev_center_c = center_c;
        prev_south_c = south_c;
    }

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_local_q] exit call=%d c0=%.9g\n",
                    call_id,
                    center_value[0]);
    }
#endif
}
