#pragma once

#include <adf.h>
#if defined(SRAD_ENABLE_AIE_VEC_HELPERS)
#include <aie_api/aie.hpp>
#endif
#include "Config.h"

using namespace adf;

using srad_local_q_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kLocalQMarginElems>>;

using srad_update_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kUpdateMarginElems>>;

using srad_mid_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

#if defined(SRAD_ENABLE_AIE_VEC_HELPERS)
namespace srad_vec {

constexpr int kCenterValuePlane = 0;
constexpr int kCenterTagPlane = 1;
constexpr int kSouthValuePlane = 2;
constexpr int kSouthTagPlane = 3;
constexpr int kChunksPerRow = srad_cfg::kRowPhysElems / srad_cfg::kLanes;
constexpr unsigned kLaneOffsets = 0x76543210;

static_assert((srad_cfg::kRowPhysElems % srad_cfg::kLanes) == 0,
              "vector path expects full physical chunks");
static_assert(srad_cfg::kLanes == 8,
              "vector path expects v8float lanes");
static_assert((kChunksPerRow & (kChunksPerRow - 1)) == 0,
              "east-neighbor chunk wrap expects a power-of-two chunk count");

inline v8float splat(float value) {
    alignas(32) float lane[srad_cfg::kLanes] = {
        value, value, value, value, value, value, value, value};
    return *(reinterpret_cast<const v8float*>(lane));
}

inline v8float load_vec(const float* __restrict base, int col) {
    return *(reinterpret_cast<const v8float*>(base + col));
}

inline void store_vec(float* __restrict base, int col, v8float value) {
    *(reinterpret_cast<v8float*>(base + col)) = value;
}

inline v8float shift_left_with_zero(v8float current,
                                    v8float previous,
                                    v8float one) {
    const v16float window = concat(previous, current);
    return fpmul(window, srad_cfg::kLanes - 1, kLaneOffsets,
                 one, 0, 0x00000000);
}

inline v8float shift_right_with_zero(v8float current,
                                     v8float next,
                                     v8float one) {
    const v16float window = concat(current, next);
    return fpmul(window, 1, kLaneOffsets, one, 0, 0x00000000);
}

inline aie::mask<srad_cfg::kLanes> make_data_lane_mask(int col) {
    alignas(32) float lane_col[srad_cfg::kLanes] = {
        static_cast<float>(col + 0),
        static_cast<float>(col + 1),
        static_cast<float>(col + 2),
        static_cast<float>(col + 3),
        static_cast<float>(col + 4),
        static_cast<float>(col + 5),
        static_cast<float>(col + 6),
        static_cast<float>(col + 7)};
    const aie::vector<float, srad_cfg::kLanes> col_v(
        *(reinterpret_cast<const v8float*>(lane_col)));
    return aie::lt(col_v, static_cast<float>(srad_cfg::kRowDataElems));
}

inline v8float select_data_lanes(
    v8float value,
    aie::mask<srad_cfg::kLanes> valid_mask,
    aie::vector<float, srad_cfg::kLanes> fill_v) {
    const aie::vector<float, srad_cfg::kLanes> value_v(value);
    return aie::select(fill_v, value_v, valid_mask).to_native();
}

}  // namespace srad_vec
#endif

extern "C" {

void srad_local_q(srad_local_q_input_buffer& in_j,
                  output_buffer<float>& out_c);

void srad_coeff_update(srad_mid_input_buffer& in_c,
                       srad_update_input_buffer& in_j,
                       output_buffer<float>& out_j_next);

}
