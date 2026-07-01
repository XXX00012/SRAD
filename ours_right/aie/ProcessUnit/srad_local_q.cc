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

struct EncodedCoeff {
    float value;
    float tag;
};

inline EncodedCoeff encode_coeff(const float* row_n,
                                 const float* row_c,
                                 const float* row_s,
                                 int col,
                                 float q0sqr) {
    if (srad_cfg::kBypassCoeffMath) {
        return {1.0f, 0.0f};
    }

    const float q0_den = q0sqr * (srad_math::kOne + q0sqr);
    if (q0_den == 0.0f) {
        return {1.0f, 0.0f};
    }

    const float jc = read_row_value(row_c, col);
    if (jc == 0.0f) {
        return {1.0f, 0.0f};
    }

    const float d_n = read_row_value(row_n, col) - jc;
    const float d_s = read_row_value(row_s, col) - jc;
    const float d_w = read_row_value(row_c, col - 1) - jc;
    const float d_e = read_row_value(row_c, col + 1) - jc;

    const float b = d_n + d_s + d_w + d_e;
    const float dq_base = jc + srad_math::kQuarter * b;
    const float dq = dq_base * dq_base;
    const float nq =
        srad_math::kHalf *
            (d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e) -
        srad_math::kOneSixteenth * b * b;
    const float num = q0_den * dq;
    const float den = nq + q0sqr * q0sqr * dq;

    EncodedCoeff encoded;
    if (den <= 0.0f || num <= 0.0f) {
        encoded.value = 0.0f;
        encoded.tag = 0.0f;
    } else if (den <= num) {
        encoded.value = 1.0f;
        encoded.tag = 0.0f;
    } else {
        encoded.value = num;
        encoded.tag = den;
    }
    return encoded;
}

inline float* plane_ptr(float* base, int plane) {
    return base + plane * srad_cfg::kMidPlaneStride;
}

inline void clear_plane_padding(float* plane) {
    for (int col = srad_cfg::kRowDataElems;
         col < srad_cfg::kMidPlaneStride;
         ++col) {
        plane[col] = 0.0f;
    }
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

    const float* base = in_j.data();

    const float* row_m_minus1 = base + 0 * srad_cfg::kRowPhysElems;
    const float* row_m = base + 1 * srad_cfg::kRowPhysElems;
    const float* row_m_plus1 = base + 2 * srad_cfg::kRowPhysElems;
    const float* row_m_plus2 = base + 3 * srad_cfg::kRowPhysElems;

    float* mid = out_c.data();
    float* center_value = plane_ptr(mid, kCenterValuePlane);
    float* center_tag = plane_ptr(mid, kCenterTagPlane);
    float* south_value = plane_ptr(mid, kSouthValuePlane);
    float* south_tag = plane_ptr(mid, kSouthTagPlane);
    float* east_value = plane_ptr(mid, kEastValuePlane);
    float* east_tag = plane_ptr(mid, kEastTagPlane);

    const float q0sqr = row_m[srad_cfg::kQ0PadIndex];
#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_local_q] q0 call=%d q0sqr=%.9g\n",
                    call_id,
                    q0sqr);
    }
#endif

    for (int col = 0; col < srad_cfg::kRowDataElems; ++col) {
        const EncodedCoeff c_center =
            encode_coeff(row_m_minus1, row_m, row_m_plus1, col, q0sqr);
        const EncodedCoeff c_south =
            encode_coeff(row_m, row_m_plus1, row_m_plus2, col, q0sqr);
        const EncodedCoeff c_east =
            encode_coeff(row_m_minus1, row_m, row_m_plus1, col + 1, q0sqr);

        center_value[col] = c_center.value;
        center_tag[col] = c_center.tag;
        south_value[col] = c_south.value;
        south_tag[col] = c_south.tag;
        east_value[col] = c_east.value;
        east_tag[col] = c_east.tag;
    }

    clear_plane_padding(center_value);
    clear_plane_padding(center_tag);
    clear_plane_padding(south_value);
    clear_plane_padding(south_tag);
    clear_plane_padding(east_value);
    clear_plane_padding(east_tag);

#if SRAD_AIE_DEBUG && (defined(__AIESIM__) || defined(__X86SIM__))
    if (call_id < 8) {
        std::printf("[srad_local_q] exit call=%d c0=%.9g\n",
                    call_id,
                    center_value[0]);
    }
#endif
}
