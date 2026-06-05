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
              "coeff-update coeff package expects center/south/east value/tag planes");
static_assert(srad_cfg::kImageInputWindowRows == srad_cfg::kInputLogicalRows,
              "coeff-update expects one full halo tile per firing");
static_assert(srad_cfg::kUpdateRowLagRows == 0,
              "coeff-update emits one full output tile per firing");
static_assert(srad_cfg::kLanes == 8,
              "coeff-update vector path expects v8float lanes");
static_assert(kChunksPerRow == 2,
              "coeff-update vector path expects two chunks per output row");
static_assert(srad_cfg::kInputRowElems == 3 * srad_cfg::kLanes,
              "coeff-update row cache expects three v8float chunks per input row");

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

inline v8float load_vec(const float* __restrict base, int idx) {
    return *(reinterpret_cast<const v8float*>(base + idx));
}

inline void store_vec(float* __restrict base, int idx, v8float value) {
    *(reinterpret_cast<v8float*>(base + idx)) = value;
}

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

inline v8float load_cached_halo_vec(const CachedJRow& row, int c) {
    if ((c & (srad_cfg::kLanes - 1)) == 0) {
        return *(reinterpret_cast<const v8float*>(row.lane + c));
    }

    alignas(32) float lane[srad_cfg::kLanes];
    for (int i = 0; i < srad_cfg::kLanes; ++i)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
        lane[i] = row.lane[c + i];
    }
    return *(reinterpret_cast<const v8float*>(lane));
}

} // namespace

extern "C" void srad_coeff_update(srad_mid_input_buffer& in_c,
                                  srad_image_input_buffer& in_j,
                                  output_buffer<float>& out_j_next) {
    const float* __restrict mid_base = in_c.data();
    const float* __restrict center_value_base =
        mid_base + kCenterValuePlaneOffset;
    const float* __restrict center_tag_base =
        mid_base + kCenterTagPlaneOffset;
    const float* __restrict south_value_base =
        mid_base + kSouthValuePlaneOffset;
    const float* __restrict south_tag_base =
        mid_base + kSouthTagPlaneOffset;
    const float* __restrict east_value_base =
        mid_base + kEastValuePlaneOffset;
    const float* __restrict east_tag_base =
        mid_base + kEastTagPlaneOffset;
    const float* __restrict j_base = in_j.data();
    float* __restrict out_base = out_j_next.data();

    if constexpr (srad_cfg::kDebugPassthroughTwoKernel) {
        const volatile float* __restrict mid_live = mid_base;
        const float keep_inputs_live = mid_live[0] - mid_live[0];

        for (int out_r = 0; out_r < srad_cfg::kOutputTileRows; ++out_r) {
            const int r = out_r + srad_cfg::kHaloTopRows;
            const float* __restrict row_c = j_base + r * COL;

            for (int out_c = 0; out_c < srad_cfg::kOutputTileCols; ++out_c)
                chess_prepare_for_pipelining
                chess_loop_range(1, ) {
                const int c = out_c + srad_cfg::kHaloLeftCols;
                out_base[out_index(out_r, out_c)] = row_c[c] + keep_inputs_live;
            }
        }
        return;
    }

    const v8float one = splat(srad_math::kOne);
    const v8float update_scale =
        splat(srad_math::kQuarter * srad_cfg::kLambdaDefault);
    const v8float zero = splat(srad_math::kZero);
    const v8float neg_one = splat(-srad_math::kOne);
    const aie::vector<float, srad_cfg::kLanes> zero_v(zero);
    const aie::vector<float, srad_cfg::kLanes> one_v(one);

    CachedJRow j_row0;
    CachedJRow j_row1;
    CachedJRow j_row2;
    CachedJRow* j_n_row = &j_row0;
    CachedJRow* j_c_row = &j_row1;
    CachedJRow* j_s_row = &j_row2;
    constexpr int first_r = srad_cfg::kHaloTopRows;
    load_j_row(j_n_row, j_base + (first_r - 1) * COL);
    load_j_row(j_c_row, j_base + first_r * COL);
    load_j_row(j_s_row, j_base + (first_r + 1) * COL);

    for (int out_r = 0; out_r < srad_cfg::kOutputTileRows; ++out_r) {
        const int r = out_r + srad_cfg::kHaloTopRows;

        for (int chunk = 0; chunk < kChunksPerRow; ++chunk)
            chess_prepare_for_pipelining
            chess_loop_range(kChunksPerRow, kChunksPerRow) {
            const int out_c = chunk * srad_cfg::kLanes;
            const int c = out_c + srad_cfg::kHaloLeftCols;
            const int out_idx = out_index(out_r, out_c);

            const v8float coeff = decode_coeff_vec(
                load_vec(center_value_base, out_idx),
                load_vec(center_tag_base, out_idx),
                zero_v, one_v);
            const v8float coeff_south = decode_coeff_vec(
                load_vec(south_value_base, out_idx),
                load_vec(south_tag_base, out_idx),
                zero_v, one_v);
            const v8float coeff_east = decode_coeff_vec(
                load_vec(east_value_base, out_idx),
                load_vec(east_tag_base, out_idx),
                zero_v, one_v);
            const v8float jc = load_cached_halo_vec(*j_c_row, c);
            const v8float jn = load_cached_halo_vec(*j_n_row, c);
            const v8float js = load_cached_halo_vec(*j_s_row, c);
            const v8float jw = load_cached_halo_vec(*j_c_row, c - 1);
            const v8float je = load_cached_halo_vec(*j_c_row, c + 1);
            const v8float dn =
                fpmac(jn, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float ds =
                fpmac(js, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float dw =
                fpmac(jw, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float de =
                fpmac(je, jc, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);

            const v8float da =
                fpmac(fpmul(coeff, 0, kLaneOffsets, dn, 0, kLaneOffsets),
                      fpmul(coeff_south, 0, kLaneOffsets, ds, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float db =
                fpmac(fpmul(coeff, 0, kLaneOffsets, dw, 0, kLaneOffsets),
                      fpmul(coeff_east, 0, kLaneOffsets, de, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float delta =
                fpmac(da, db, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float out =
                fpmac(jc,
                      fpmul(update_scale, 0, kLaneOffsets, delta, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            store_vec(out_base, out_idx, out);
        }

        if (out_r != srad_cfg::kOutputTileRows - 1) {
            CachedJRow* recycle_row = j_n_row;
            j_n_row = j_c_row;
            j_c_row = j_s_row;
            j_s_row = recycle_row;
            load_j_row(j_s_row, j_base + (r + 2) * COL);
        }
    }
}
