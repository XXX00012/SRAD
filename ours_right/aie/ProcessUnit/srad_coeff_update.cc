#include <adf.h>
#ifndef SRAD_AIE_DEBUG
#define SRAD_AIE_DEBUG 0
#endif

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
#include <cstdio>
#endif

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

namespace {

constexpr int kCenterValuePlane = 0;
constexpr int kCenterTagPlane = 1;
constexpr int kSouthValuePlane = 2;
constexpr int kSouthTagPlane = 3;
constexpr int kEastValuePlane = 4;
constexpr int kEastTagPlane = 5;

inline float read_row_value(const float* row, int col) {
    if (col < 0 || col >= srad_cfg::kRowDataElems) {
        return 0.0f;
    }
    return row[col];
}

inline const float* plane_ptr(const float* base, int plane) {
    return base + plane * srad_cfg::kMidPlaneStride;
}

inline float decode_coeff(float value, float tag) {
    return (tag > 0.0f) ? (value / tag) : value;
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

    const float* mid = in_c.data();
    const float* center_value = plane_ptr(mid, kCenterValuePlane);
    const float* center_tag = plane_ptr(mid, kCenterTagPlane);
    const float* south_value = plane_ptr(mid, kSouthValuePlane);
    const float* south_tag = plane_ptr(mid, kSouthTagPlane);
    const float* east_value = plane_ptr(mid, kEastValuePlane);
    const float* east_tag = plane_ptr(mid, kEastTagPlane);

    const float* base = in_j.data();
    const float* row_m_minus1 = base + 0 * srad_cfg::kRowPhysElems;
    const float* row_m = base + 1 * srad_cfg::kRowPhysElems;
    const float* row_m_plus1 = base + 2 * srad_cfg::kRowPhysElems;

    float* out = out_j_next.data();
    float row_sum = 0.0f;
    float row_sum2 = 0.0f;

    for (int col = 0; col < srad_cfg::kRowDataElems; ++col) {
        const float jc = read_row_value(row_m, col);
        const float d_n = read_row_value(row_m_minus1, col) - jc;
        const float d_s = read_row_value(row_m_plus1, col) - jc;
        const float d_w = read_row_value(row_m, col - 1) - jc;
        const float d_e = read_row_value(row_m, col + 1) - jc;

        const float coeff =
            decode_coeff(center_value[col], center_tag[col]);
        const float coeff_south =
            decode_coeff(south_value[col], south_tag[col]);
        const float coeff_east =
            decode_coeff(east_value[col], east_tag[col]);

        const float divergence =
            coeff * d_n + coeff_south * d_s +
            coeff * d_w + coeff_east * d_e;
        const float next =
            jc + srad_math::kQuarter * srad_cfg::kLambdaDefault * divergence;

        out[col] = next;
        row_sum += next;
        row_sum2 += next * next;
    }

    for (int col = srad_cfg::kRowDataElems;
         col < srad_cfg::kRowPhysElems;
         ++col) {
        out[col] = 0.0f;
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
