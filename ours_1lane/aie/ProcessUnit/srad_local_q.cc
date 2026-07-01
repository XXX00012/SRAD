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
                             float q0sqr,
                             v8float q0_num_factor,
                             v8float q0sqr2,
                             v8float zero,
                             v8float one,
                             v8float neg_one,
                             v8float quarter,
                             v8float half,
                             v8float one_sixteenth,
                             aie::vector<float, srad_cfg::kLanes> zero_v,
                             aie::vector<float, srad_cfg::kLanes> one_v,
                             v8float* value_out,
                             v8float* tag_out) {
    if constexpr (srad_cfg::kBypassCoeffMath) {
        *value_out = one;
        *tag_out = zero;
        return;
    }
    if (q0sqr <= srad_math::kZero) {
        *value_out = one;
        *tag_out = zero;
        return;
    }

    const v8float d_n =
        fpmac(j_n, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_s =
        fpmac(j_s, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_w =
        fpmac(j_w, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_e =
        fpmac(j_e, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

    const v8float grad2_ns =
        fpmac(fpmul(d_n, 0, kLaneOffsets, d_n, 0, kLaneOffsets),
              fpmul(d_s, 0, kLaneOffsets, d_s, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float grad2_we =
        fpmac(fpmul(d_w, 0, kLaneOffsets, d_w, 0, kLaneOffsets),
              fpmul(d_e, 0, kLaneOffsets, d_e, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float grad2_sum =
        fpmac(grad2_ns, grad2_we, 0, kLaneOffsets, one, 0, kLaneOffsets);

    const v8float b_ns =
        fpmac(d_n, d_s, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float b_we =
        fpmac(d_w, d_e, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float b =
        fpmac(b_ns, b_we, 0, kLaneOffsets, one, 0, kLaneOffsets);

    const aie::vector<float, srad_cfg::kLanes> jc_v(j_c);
    const auto center_zero_mask = aie::le(jc_v, zero_v);

    const v8float b2 = fpmul(b, 0, kLaneOffsets, b, 0, kLaneOffsets);
    const v8float dq_base =
        fpmac(j_c,
              fpmul(quarter, 0, kLaneOffsets, b, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float dq =
        fpmul(dq_base, 0, kLaneOffsets, dq_base, 0, kLaneOffsets);
    const v8float nq =
        fpmac(fpmul(half, 0, kLaneOffsets, grad2_sum, 0, kLaneOffsets),
              fpmul(one_sixteenth, 0, kLaneOffsets, b2, 0, kLaneOffsets),
              0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const aie::vector<float, srad_cfg::kLanes> q0_num_v(q0_num_factor);
    const aie::vector<float, srad_cfg::kLanes> q0_den_v(
        fpmac(nq,
              fpmul(q0sqr2, 0, kLaneOffsets, dq, 0, kLaneOffsets),
              0, kLaneOffsets, one, 0, kLaneOffsets));
    const auto q0_den_zero_mask = aie::eq(q0_den_v, zero_v);
    const auto safe_q0_den_v = aie::select(q0_den_v, one_v, q0_den_zero_mask);
    const auto c_v =
        aie::mul(q0_num_v, aie::inv(safe_q0_den_v)).template to_vector<float>();

    auto value_v = aie::select(c_v, one_v, q0_den_zero_mask);
    value_v = aie::select(value_v, one_v, center_zero_mask);
    auto tag_v = aie::select(one_v, zero_v, q0_den_zero_mask);
    tag_v = aie::select(tag_v, zero_v, center_zero_mask);

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
    const float q0sqr = row_m[srad_cfg::kQ0PadIndex];
    const float q0sqr2_scalar = q0sqr * q0sqr;
    const float q0_num_factor_scalar =
        q0sqr * (srad_math::kOne + q0sqr);

    float* __restrict mid = out_c.data();
    float* __restrict center_value =
        mid + kCenterValuePlane * srad_cfg::kMidPlaneStride;
    float* __restrict center_tag =
        mid + kCenterTagPlane * srad_cfg::kMidPlaneStride;
    float* __restrict south_value =
        mid + kSouthValuePlane * srad_cfg::kMidPlaneStride;
    float* __restrict south_tag =
        mid + kSouthTagPlane * srad_cfg::kMidPlaneStride;

    const v8float zero = splat(srad_math::kZero);
    const v8float one = splat(srad_math::kOne);
    const v8float neg_one = splat(-srad_math::kOne);
    const v8float quarter = splat(srad_math::kQuarter);
    const v8float half = splat(srad_math::kHalf);
    const v8float one_sixteenth = splat(srad_math::kOneSixteenth);
    const v8float q0sqr2 = splat(q0sqr2_scalar);
    const v8float q0_num_factor = splat(q0_num_factor_scalar);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);

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
                         q0sqr, q0_num_factor, q0sqr2,
                         zero, one, neg_one, quarter, half, one_sixteenth,
                         zero_v, one_v,
                         &center_value_vec, &center_tag_vec);

        v8float south_value_vec;
        v8float south_tag_vec;
        encode_coeff_vec(south_c, south_n, south_s, south_w, south_e,
                         q0sqr, q0_num_factor, q0sqr2,
                         zero, one, neg_one, quarter, half, one_sixteenth,
                         zero_v, one_v,
                         &south_value_vec, &south_tag_vec);

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
