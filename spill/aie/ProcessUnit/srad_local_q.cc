#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

namespace {

constexpr int kCenterValuePlaneOffset = 0 * srad_cfg::kOutputTilePixels;
constexpr int kCenterTagPlaneOffset = 1 * srad_cfg::kOutputTilePixels;
constexpr int kSouthValuePlaneOffset = 2 * srad_cfg::kOutputTilePixels;
constexpr int kSouthTagPlaneOffset = 3 * srad_cfg::kOutputTilePixels;
constexpr int kEastValuePlaneOffset = 4 * srad_cfg::kOutputTilePixels;
constexpr int kEastTagPlaneOffset = 5 * srad_cfg::kOutputTilePixels;
constexpr unsigned kLaneOffsets = 0x76543210;
constexpr int kChunksPerRow = srad_cfg::kOutputTileCols / srad_cfg::kLanes;

static_assert(srad_cfg::kMidRecordElems == 6,
              "local-q coeff package expects center/south/east value/tag planes");
static_assert(srad_cfg::kImageInputWindowRows == srad_cfg::kInputLogicalRows,
              "local-q expects one full halo tile per firing");
static_assert(srad_cfg::kUpdateRowLagRows == 0,
              "local-q emits one full output tile per firing");
static_assert(srad_cfg::kLanes == 8,
              "local-q vector path expects v8float lanes");
static_assert(kChunksPerRow == 2,
              "local-q vector path expects two chunks per output row");
static_assert(srad_cfg::kInputRowElems == 3 * srad_cfg::kLanes,
              "local-q row cache expects three v8float chunks per input row");

inline int west_col_local(int c) {
    return (c == 0) ? 0 : c - 1;
}

inline int east_col_local(int c) {
    return (c == COL - 1) ? COL - 1 : c + 1;
}

inline int out_index(int r, int c) {
    return r * srad_cfg::kOutputTileCols + c;
}

inline v8float splat(float value) {
    alignas(32) float lane[srad_cfg::kLanes] = {
        value, value, value, value, value, value, value, value};
    return *(reinterpret_cast<const v8float*>(lane));
}

inline v8float load_row_chunk(const float* __restrict row_base, int chunk) {
    return *(reinterpret_cast<const v8float*>(
        row_base + chunk * srad_cfg::kLanes));
}

struct alignas(32) CachedJRow {
    float lane[srad_cfg::kInputRowElems];
};

inline void load_j_row(CachedJRow* __restrict row,
                       const float* __restrict row_base) {
    *(reinterpret_cast<v8float*>(row->lane + 0 * srad_cfg::kLanes)) =
        load_row_chunk(row_base, 0);
    *(reinterpret_cast<v8float*>(row->lane + 1 * srad_cfg::kLanes)) =
        load_row_chunk(row_base, 1);
    *(reinterpret_cast<v8float*>(row->lane + 2 * srad_cfg::kLanes)) =
        load_row_chunk(row_base, 2);
}

template <int C>
inline v8float load_cached_halo_vec(const CachedJRow& row) {
    static_assert(C >= 0, "cached vector starts before the row");
    static_assert(C + srad_cfg::kLanes <= srad_cfg::kInputRowElems,
                  "cached vector extends past the row");

    if constexpr ((C & (srad_cfg::kLanes - 1)) == 0) {
        return *(reinterpret_cast<const v8float*>(row.lane + C));
    }

    alignas(32) float lane[srad_cfg::kLanes];
    for (int i = 0; i < srad_cfg::kLanes; ++i)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
        lane[i] = row.lane[C + i];
    }
    return *(reinterpret_cast<const v8float*>(lane));
}

inline void store_vec(float* __restrict base, int idx, v8float value) {
    *(reinterpret_cast<v8float*>(base + idx)) = value;
}

struct EncodedCoeffScalar {
    float value;
    float tag;
};

inline EncodedCoeffScalar encode_coeff_scalar_fast(float jc,
                                                   float d_n,
                                                   float d_s,
                                                   float d_w,
                                                   float d_e,
                                                   float q0sqr2,
                                                   float q0_den) {
    if constexpr (srad_cfg::kBypassCoeffMath) {
        return {srad_math::kOne, srad_math::kZero};
    }

    if (jc == srad_math::kZero || q0_den == srad_math::kZero) {
        return {srad_math::kOne, srad_math::kZero};
    }

    const float a = d_n * d_n + d_s * d_s + d_w * d_w + d_e * d_e;
    const float b = d_n + d_s + d_w + d_e;
    const float n =
        srad_math::kHalf * a - srad_math::kOneSixteenth * b * b;
    const float t = jc + srad_math::kQuarter * b;
    const float d = t * t;
    const float num = q0_den * d;
    const float den = n + q0sqr2 * d;

    if (den <= srad_math::kZero || num <= srad_math::kZero) {
        return {srad_math::kZero, srad_math::kZero};
    }
    if (den <= num) {
        return {srad_math::kOne, srad_math::kZero};
    }
    return {num, den};
}

inline EncodedCoeffScalar encode_coeff_scalar_at(
    const CachedJRow& row_n,
    const CachedJRow& row_c,
    const CachedJRow& row_s,
    int c,
    float q0sqr2,
    float q0_den) {
    const float jc = row_c.lane[c];
    const float d_n = row_n.lane[c] - jc;
    const float d_s = row_s.lane[c] - jc;
    const float d_w = row_c.lane[west_col_local(c)] - jc;
    const float d_e = row_c.lane[east_col_local(c)] - jc;

    return encode_coeff_scalar_fast(jc, d_n, d_s, d_w, d_e,
                                    q0sqr2, q0_den);
}

inline void encode_coeff_vec(v8float j_c,
                             v8float num,
                             v8float den,
                             v8float zero,
                             v8float one,
                             aie::vector<float, srad_cfg::kLanes> zero_v,
                             aie::vector<float, srad_cfg::kLanes> one_v,
                             v8float* value_out,
                             v8float* tag_out) {
    const aie::vector<float, srad_cfg::kLanes> jc_v(j_c);
    const aie::vector<float, srad_cfg::kLanes> num_v(num);
    const aie::vector<float, srad_cfg::kLanes> den_v(den);

    const auto zero_mask = aie::le(den_v, srad_math::kZero);
    const auto num_zero_mask = aie::le(num_v, srad_math::kZero);
    const auto one_mask = aie::le(den_v, num_v);
    const auto center_zero_mask = aie::le(jc_v, srad_math::kZero);

    auto value_v = aie::select(num_v, one_v, one_mask);
    value_v = aie::select(value_v, zero_v, zero_mask);
    value_v = aie::select(value_v, zero_v, num_zero_mask);
    value_v = aie::select(value_v, one_v, center_zero_mask);

    auto tag_v = aie::select(den_v, zero_v, one_mask);
    tag_v = aie::select(tag_v, zero_v, zero_mask);
    tag_v = aie::select(tag_v, zero_v, num_zero_mask);
    tag_v = aie::select(tag_v, zero_v, center_zero_mask);

    *value_out = value_v.to_native();
    *tag_out = tag_v.to_native();
}

template <int C>
inline void compute_coeff_vec_at(const CachedJRow& row_n,
                                 const CachedJRow& row_c,
                                 const CachedJRow& row_s,
                                 float q0_den_scalar,
                                 v8float q0sqr2,
                                 v8float q0_den,
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
    static_assert(C > 0, "coefficient vector needs a west halo");
    static_assert(C + srad_cfg::kLanes < srad_cfg::kInputRowElems,
                  "coefficient vector needs an east halo");

    const v8float j_c = load_cached_halo_vec<C>(row_c);
    const v8float j_n = load_cached_halo_vec<C>(row_n);
    const v8float j_s = load_cached_halo_vec<C>(row_s);
    const v8float j_w = load_cached_halo_vec<C - 1>(row_c);
    const v8float j_e = load_cached_halo_vec<C + 1>(row_c);

    const v8float d_n =
        fpmac(j_n, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_s =
        fpmac(j_s, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_w =
        fpmac(j_w, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
    const v8float d_e =
        fpmac(j_e, j_c, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

    if constexpr (srad_cfg::kBypassCoeffMath) {
        *value_out = one;
        *tag_out = zero;
        return;
    }
    if (q0_den_scalar == srad_math::kZero) {
        *value_out = one;
        *tag_out = zero;
        return;
    }

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

    encode_coeff_vec(j_c, num, den, zero, one, zero_v, one_v,
                     value_out, tag_out);
}

} // namespace

extern "C" void srad_local_q(srad_image_input_buffer& in_j,
                             output_buffer<float>& out_c) {
    const float* __restrict j_base = in_j.data();
    float* __restrict mid_base = out_c.data();

    float* __restrict center_value_base = mid_base + kCenterValuePlaneOffset;
    float* __restrict center_tag_base = mid_base + kCenterTagPlaneOffset;
    float* __restrict south_value_base = mid_base + kSouthValuePlaneOffset;
    float* __restrict south_tag_base = mid_base + kSouthTagPlaneOffset;
    float* __restrict east_value_base = mid_base + kEastValuePlaneOffset;
    float* __restrict east_tag_base = mid_base + kEastTagPlaneOffset;

    if constexpr (srad_cfg::kDebugPassthroughTwoKernel) {
        const volatile float* __restrict j_live = j_base;
        const float keep_inputs_live =
            (j_live[0] - j_live[0]) +
            (j_live[srad_cfg::kQ0SqrTileIndex] -
             j_live[srad_cfg::kQ0SqrTileIndex]);

        for (int out_r = 0; out_r < srad_cfg::kOutputTileRows; ++out_r) {
            for (int out_c = 0; out_c < srad_cfg::kOutputTileCols; ++out_c)
                chess_prepare_for_pipelining
                chess_loop_range(1, ) {
                const int out_idx = out_index(out_r, out_c);
                center_value_base[out_idx] = srad_math::kOne + keep_inputs_live;
                center_tag_base[out_idx] = srad_math::kZero;
                south_value_base[out_idx] = srad_math::kOne;
                south_tag_base[out_idx] = srad_math::kZero;
                east_value_base[out_idx] = srad_math::kOne;
                east_tag_base[out_idx] = srad_math::kZero;
            }
        }
        return;
    }

    const float q0sqr = j_base[srad_cfg::kQ0SqrTileIndex];
    const float q0sqr2_scalar = q0sqr * q0sqr;
    const float q0_den_scalar = q0sqr * (srad_math::kOne + q0sqr);

    const v8float one = splat(srad_math::kOne);
    const v8float zero = splat(srad_math::kZero);
    const v8float neg_one = splat(-srad_math::kOne);
    const v8float quarter = splat(srad_math::kQuarter);
    const v8float half = splat(srad_math::kHalf);
    const v8float one_sixteenth = splat(srad_math::kOneSixteenth);
    const v8float q0 = splat(q0sqr);
    const v8float q0sqr2 =
        fpmul(q0, 0, kLaneOffsets, q0, 0, kLaneOffsets);
    const v8float one_plus_q0 =
        fpmac(one, q0, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float q0_den =
        fpmul(q0, 0, kLaneOffsets, one_plus_q0, 0, kLaneOffsets);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);

    constexpr int c0 = srad_cfg::kHaloLeftCols;
    constexpr int c1 = srad_cfg::kHaloLeftCols + srad_cfg::kLanes;
    const int c_east_tail =
        srad_cfg::kHaloLeftCols + srad_cfg::kOutputTileCols;
    const int initial_south_r =
        srad_cfg::kHaloTopRows + srad_cfg::kOutputTileRows;

    CachedJRow j_row0;
    CachedJRow j_row1;
    CachedJRow j_row2;
    CachedJRow* j_n_row = &j_row0;
    CachedJRow* j_c_row = &j_row1;
    CachedJRow* j_s_row = &j_row2;
    load_j_row(j_n_row, j_base + (initial_south_r - 1) * COL);
    load_j_row(j_c_row, j_base + initial_south_r * COL);
    load_j_row(j_s_row, j_base + (initial_south_r + 1) * COL);

    v8float south_value0;
    v8float south_tag0;
    v8float south_value1;
    v8float south_tag1;
    compute_coeff_vec_at<c0>(*j_n_row, *j_c_row, *j_s_row,
                             q0_den_scalar, q0sqr2, q0_den,
                             zero, one, neg_one, quarter, half,
                             one_sixteenth, zero_v, one_v,
                             &south_value0, &south_tag0);
    compute_coeff_vec_at<c1>(*j_n_row, *j_c_row, *j_s_row,
                             q0_den_scalar, q0sqr2, q0_den,
                             zero, one, neg_one, quarter, half,
                             one_sixteenth, zero_v, one_v,
                             &south_value1, &south_tag1);

    for (int out_r = srad_cfg::kOutputTileRows - 1; out_r >= 0; --out_r) {
        const int r = out_r + srad_cfg::kHaloTopRows;
        const int out_idx0 = out_index(out_r, 0);
        const int out_idx1 = out_index(out_r, srad_cfg::kLanes);

        CachedJRow* recycle_row = j_s_row;
        j_s_row = j_c_row;
        j_c_row = j_n_row;
        j_n_row = recycle_row;
        load_j_row(j_n_row, j_base + (r - 1) * COL);

        v8float row_value0;
        v8float row_tag0;
        v8float row_value1;
        v8float row_tag1;
        compute_coeff_vec_at<c0>(*j_n_row, *j_c_row, *j_s_row,
                                 q0_den_scalar, q0sqr2, q0_den,
                                 zero, one, neg_one, quarter, half,
                                 one_sixteenth, zero_v, one_v,
                                 &row_value0, &row_tag0);
        compute_coeff_vec_at<c1>(*j_n_row, *j_c_row, *j_s_row,
                                 q0_den_scalar, q0sqr2, q0_den,
                                 zero, one, neg_one, quarter, half,
                                 one_sixteenth, zero_v, one_v,
                                 &row_value1, &row_tag1);
        const EncodedCoeffScalar east_tail =
            encode_coeff_scalar_at(*j_n_row, *j_c_row, *j_s_row,
                                   c_east_tail, q0sqr2_scalar,
                                   q0_den_scalar);

        store_vec(center_value_base, out_idx0, row_value0);
        store_vec(center_tag_base, out_idx0, row_tag0);
        store_vec(center_value_base, out_idx1, row_value1);
        store_vec(center_tag_base, out_idx1, row_tag1);
        store_vec(south_value_base, out_idx0, south_value0);
        store_vec(south_tag_base, out_idx0, south_tag0);
        store_vec(south_value_base, out_idx1, south_value1);
        store_vec(south_tag_base, out_idx1, south_tag1);

        alignas(32) float row_value0_lane[srad_cfg::kLanes];
        alignas(32) float row_tag0_lane[srad_cfg::kLanes];
        alignas(32) float row_value1_lane[srad_cfg::kLanes];
        alignas(32) float row_tag1_lane[srad_cfg::kLanes];
        alignas(32) float east_value0_lane[srad_cfg::kLanes];
        alignas(32) float east_tag0_lane[srad_cfg::kLanes];
        alignas(32) float east_value1_lane[srad_cfg::kLanes];
        alignas(32) float east_tag1_lane[srad_cfg::kLanes];
        *(reinterpret_cast<v8float*>(row_value0_lane)) = row_value0;
        *(reinterpret_cast<v8float*>(row_tag0_lane)) = row_tag0;
        *(reinterpret_cast<v8float*>(row_value1_lane)) = row_value1;
        *(reinterpret_cast<v8float*>(row_tag1_lane)) = row_tag1;
        for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
            east_value0_lane[lane] =
                (lane == srad_cfg::kLanes - 1) ?
                    row_value1_lane[0] :
                    row_value0_lane[lane + 1];
            east_tag0_lane[lane] =
                (lane == srad_cfg::kLanes - 1) ?
                    row_tag1_lane[0] :
                    row_tag0_lane[lane + 1];
            east_value1_lane[lane] =
                (lane == srad_cfg::kLanes - 1) ?
                    east_tail.value :
                    row_value1_lane[lane + 1];
            east_tag1_lane[lane] =
                (lane == srad_cfg::kLanes - 1) ?
                    east_tail.tag :
                    row_tag1_lane[lane + 1];
        }

        store_vec(east_value_base, out_idx0,
                  *(reinterpret_cast<const v8float*>(east_value0_lane)));
        store_vec(east_tag_base, out_idx0,
                  *(reinterpret_cast<const v8float*>(east_tag0_lane)));
        store_vec(east_value_base, out_idx1,
                  *(reinterpret_cast<const v8float*>(east_value1_lane)));
        store_vec(east_tag_base, out_idx1,
                  *(reinterpret_cast<const v8float*>(east_tag1_lane)));

        south_value0 = row_value0;
        south_tag0 = row_tag0;
        south_value1 = row_value1;
        south_tag1 = row_tag1;
    }
}
