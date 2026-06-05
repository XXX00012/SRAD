#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

namespace {

constexpr unsigned kLaneOffsets = 0x76543210;
constexpr int kChunksPerRow = BLOCK_COL / srad_cfg::kLanes;

static_assert(BLOCK_ROW == 16, "single-row SRAD fused kernel expects 16 rows");
static_assert(BLOCK_COL == 16, "single-row SRAD fused kernel expects 16 cols");
static_assert(kChunksPerRow == 2,
              "single-row SRAD fused kernel processes two v8float chunks");

} // namespace

void srad_undivide_fused(srad_image_input_buffer& in_j,
                         srad_image_input_buffer& in_j_update,
                         srad_scalar_input_buffer& in_q0sqr,
                         srad_scalar_input_buffer& in_lambda,
                         output_buffer<float>& out_j_next) {
    const float* __restrict j_base = in_j.data();
    const float* __restrict j_update_base = in_j_update.data();
    const float* __restrict q0_base = in_q0sqr.data();
    const float* __restrict lambda_base = in_lambda.data();
    float* __restrict out_base = out_j_next.data();

    const float q0sqr = q0_base[0];
    const float lambda = lambda_base[0];

    alignas(32) float q0_lane[8] = {
        q0sqr, q0sqr, q0sqr, q0sqr, q0sqr, q0sqr, q0sqr, q0sqr};
    alignas(32) float one_lane[8] = {
        srad_math::kOne, srad_math::kOne, srad_math::kOne, srad_math::kOne,
        srad_math::kOne, srad_math::kOne, srad_math::kOne, srad_math::kOne};
    alignas(32) float neg_one_lane[8] = {
        -srad_math::kOne, -srad_math::kOne, -srad_math::kOne,
        -srad_math::kOne, -srad_math::kOne, -srad_math::kOne,
        -srad_math::kOne, -srad_math::kOne};
    alignas(32) float quarter_lane[8] = {
        srad_math::kQuarter, srad_math::kQuarter, srad_math::kQuarter,
        srad_math::kQuarter, srad_math::kQuarter, srad_math::kQuarter,
        srad_math::kQuarter, srad_math::kQuarter};
    alignas(32) float half_lane[8] = {
        srad_math::kHalf, srad_math::kHalf, srad_math::kHalf,
        srad_math::kHalf, srad_math::kHalf, srad_math::kHalf,
        srad_math::kHalf, srad_math::kHalf};
    alignas(32) float sixteenth_lane[8] = {
        srad_math::kOneSixteenth, srad_math::kOneSixteenth,
        srad_math::kOneSixteenth, srad_math::kOneSixteenth,
        srad_math::kOneSixteenth, srad_math::kOneSixteenth,
        srad_math::kOneSixteenth, srad_math::kOneSixteenth};
    alignas(32) float update_lane[8] = {
        srad_math::kQuarter * lambda, srad_math::kQuarter * lambda,
        srad_math::kQuarter * lambda, srad_math::kQuarter * lambda,
        srad_math::kQuarter * lambda, srad_math::kQuarter * lambda,
        srad_math::kQuarter * lambda, srad_math::kQuarter * lambda};

    const v8float q0 = *(reinterpret_cast<const v8float*>(q0_lane));
    const v8float one = *(reinterpret_cast<const v8float*>(one_lane));
    const v8float neg_one = *(reinterpret_cast<const v8float*>(neg_one_lane));
    const v8float quarter = *(reinterpret_cast<const v8float*>(quarter_lane));
    const v8float half = *(reinterpret_cast<const v8float*>(half_lane));
    const v8float one_sixteenth =
        *(reinterpret_cast<const v8float*>(sixteenth_lane));
    const v8float update_scale =
        *(reinterpret_cast<const v8float*>(update_lane));
    const v8float q0sqr2 =
        fpmul(q0, 0, kLaneOffsets, q0, 0, kLaneOffsets);
    const v8float one_plus_q0 =
        fpmac(one, q0, 0, kLaneOffsets, one, 0, kLaneOffsets);
    const v8float q0_den =
        fpmul(q0, 0, kLaneOffsets, one_plus_q0, 0, kLaneOffsets);

    alignas(32) float row_j[BLOCK_COL];
    alignas(32) float row_w[srad_cfg::kLanes];
    alignas(32) float row_e[srad_cfg::kLanes];
    alignas(32) float c_row[BLOCK_COL];
    alignas(32) float c_south_row[BLOCK_COL];
    alignas(32) float c_east[srad_cfg::kLanes];
    alignas(32) float num_lane[srad_cfg::kLanes];
    alignas(32) float den_lane[srad_cfg::kLanes];
    alignas(32) float c_lane[srad_cfg::kLanes];

    for (int row = BLOCK_ROW - 1; row >= 0; --row) {
        const int row_n = (row == 0) ? 0 : row - 1;
        const int row_s = (row == BLOCK_ROW - 1) ? BLOCK_ROW - 1 : row + 1;

        const int base_n = row_n * BLOCK_COL;
        const int base_c = row * BLOCK_COL;
        const int base_s = row_s * BLOCK_COL;

        const v8float j_c0 =
            *(reinterpret_cast<const v8float*>(j_base + base_c));
        const v8float j_c1 =
            *(reinterpret_cast<const v8float*>(j_base + base_c + 8));
        *(reinterpret_cast<v8float*>(row_j)) = j_c0;
        *(reinterpret_cast<v8float*>(row_j + 8)) = j_c1;

        const v8float j_n0 =
            *(reinterpret_cast<const v8float*>(j_base + base_n));
        const v8float j_s0 =
            *(reinterpret_cast<const v8float*>(j_base + base_s));

        v8float c0;
        if constexpr (srad_cfg::kBypassCoeffMath) {
            c0 = one;
        } else {
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    const int col = lane;
                    const int west_col = (col == 0) ? 0 : col - 1;
                    const int east_col = col + 1;
                    row_w[lane] = row_j[west_col];
                    row_e[lane] = row_j[east_col];
                }

            const v8float w0 =
                *(reinterpret_cast<const v8float*>(row_w));
            const v8float e0 =
                *(reinterpret_cast<const v8float*>(row_e));
            const v8float d_n0 =
                fpmac(j_n0, j_c0, 0, kLaneOffsets, neg_one, 0,
                      kLaneOffsets);
            const v8float d_s0 =
                fpmac(j_s0, j_c0, 0, kLaneOffsets, neg_one, 0,
                      kLaneOffsets);
            const v8float d_w0 =
                fpmac(w0, j_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float d_e0 =
                fpmac(e0, j_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float a0_0 =
                fpmac(fpmul(d_n0, 0, kLaneOffsets, d_n0, 0, kLaneOffsets),
                      fpmul(d_s0, 0, kLaneOffsets, d_s0, 0, kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float a1_0 =
                fpmac(fpmul(d_w0, 0, kLaneOffsets, d_w0, 0, kLaneOffsets),
                      fpmul(d_e0, 0, kLaneOffsets, d_e0, 0, kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float a_0 =
                fpmac(a0_0, a1_0, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b0_0 =
                fpmac(d_n0, d_s0, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b1_0 =
                fpmac(d_w0, d_e0, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b_0 =
                fpmac(b0_0, b1_0, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float n_0 =
                fpmac(fpmul(half, 0, kLaneOffsets, a_0, 0, kLaneOffsets),
                      fpmul(one_sixteenth, 0, kLaneOffsets,
                            fpmul(b_0, 0, kLaneOffsets, b_0, 0,
                                  kLaneOffsets),
                            0, kLaneOffsets),
                      0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float t_0 =
                fpmac(j_c0,
                      fpmul(quarter, 0, kLaneOffsets, b_0, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float d_0 =
                fpmul(t_0, 0, kLaneOffsets, t_0, 0, kLaneOffsets);
            const v8float den0 =
                fpmac(n_0,
                      fpmul(q0sqr2, 0, kLaneOffsets, d_0, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float num0 =
                fpmul(q0_den, 0, kLaneOffsets, d_0, 0, kLaneOffsets);
            *(reinterpret_cast<v8float*>(num_lane)) = num0;
            *(reinterpret_cast<v8float*>(den_lane)) = den0;
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    if (den_lane[lane] <= srad_math::kZero) {
                        c_lane[lane] = srad_math::kZero;
                    } else if (den_lane[lane] < num_lane[lane]) {
                        c_lane[lane] = srad_math::kOne;
                    } else {
                        c_lane[lane] = num_lane[lane] / den_lane[lane];
                    }
                }
            c0 = *(reinterpret_cast<const v8float*>(c_lane));
        }
        *(reinterpret_cast<v8float*>(c_row)) = c0;

        const v8float j_n1 =
            *(reinterpret_cast<const v8float*>(j_base + base_n + 8));
        const v8float j_s1 =
            *(reinterpret_cast<const v8float*>(j_base + base_s + 8));

        v8float c1;
        if constexpr (srad_cfg::kBypassCoeffMath) {
            c1 = one;
        } else {
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    const int col = 8 + lane;
                    const int west_col = col - 1;
                    const int east_col =
                        (col == BLOCK_COL - 1) ? BLOCK_COL - 1 : col + 1;
                    row_w[lane] = row_j[west_col];
                    row_e[lane] = row_j[east_col];
                }

            const v8float w1 =
                *(reinterpret_cast<const v8float*>(row_w));
            const v8float e1 =
                *(reinterpret_cast<const v8float*>(row_e));
            const v8float d_n1 =
                fpmac(j_n1, j_c1, 0, kLaneOffsets, neg_one, 0,
                      kLaneOffsets);
            const v8float d_s1 =
                fpmac(j_s1, j_c1, 0, kLaneOffsets, neg_one, 0,
                      kLaneOffsets);
            const v8float d_w1 =
                fpmac(w1, j_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float d_e1 =
                fpmac(e1, j_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float a0_1 =
                fpmac(fpmul(d_n1, 0, kLaneOffsets, d_n1, 0, kLaneOffsets),
                      fpmul(d_s1, 0, kLaneOffsets, d_s1, 0, kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float a1_1 =
                fpmac(fpmul(d_w1, 0, kLaneOffsets, d_w1, 0, kLaneOffsets),
                      fpmul(d_e1, 0, kLaneOffsets, d_e1, 0, kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float a_1 =
                fpmac(a0_1, a1_1, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b0_1 =
                fpmac(d_n1, d_s1, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b1_1 =
                fpmac(d_w1, d_e1, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float b_1 =
                fpmac(b0_1, b1_1, 0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float n_1 =
                fpmac(fpmul(half, 0, kLaneOffsets, a_1, 0, kLaneOffsets),
                      fpmul(one_sixteenth, 0, kLaneOffsets,
                            fpmul(b_1, 0, kLaneOffsets, b_1, 0,
                                  kLaneOffsets),
                            0, kLaneOffsets),
                      0, kLaneOffsets, neg_one, 0, kLaneOffsets);
            const v8float t_1 =
                fpmac(j_c1,
                      fpmul(quarter, 0, kLaneOffsets, b_1, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float d_1 =
                fpmul(t_1, 0, kLaneOffsets, t_1, 0, kLaneOffsets);
            const v8float den1 =
                fpmac(n_1,
                      fpmul(q0sqr2, 0, kLaneOffsets, d_1, 0,
                            kLaneOffsets),
                      0, kLaneOffsets, one, 0, kLaneOffsets);
            const v8float num1 =
                fpmul(q0_den, 0, kLaneOffsets, d_1, 0, kLaneOffsets);
            *(reinterpret_cast<v8float*>(num_lane)) = num1;
            *(reinterpret_cast<v8float*>(den_lane)) = den1;
            for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
                chess_prepare_for_pipelining
                chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                    if (den_lane[lane] <= srad_math::kZero) {
                        c_lane[lane] = srad_math::kZero;
                    } else if (den_lane[lane] < num_lane[lane]) {
                        c_lane[lane] = srad_math::kOne;
                    } else {
                        c_lane[lane] = num_lane[lane] / den_lane[lane];
                    }
                }
            c1 = *(reinterpret_cast<const v8float*>(c_lane));
        }
        *(reinterpret_cast<v8float*>(c_row + 8)) = c1;

        v8float c_s0;
        v8float c_s1;
        if (row == BLOCK_ROW - 1) {
            c_s0 = c0;
            c_s1 = c1;
        } else {
            c_s0 = *(reinterpret_cast<const v8float*>(c_south_row));
            c_s1 = *(reinterpret_cast<const v8float*>(c_south_row + 8));
        }

        const v8float u_c0 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_c));
        const v8float u_c1 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_c + 8));
        *(reinterpret_cast<v8float*>(row_j)) = u_c0;
        *(reinterpret_cast<v8float*>(row_j + 8)) = u_c1;

        const v8float u_n0 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_n));
        const v8float u_s0 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_s));

        for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                const int col = lane;
                const int west_col = (col == 0) ? 0 : col - 1;
                const int east_col = col + 1;
                row_w[lane] = row_j[west_col];
                row_e[lane] = row_j[east_col];
                c_east[lane] = c_row[east_col];
            }

        const v8float uw0 =
            *(reinterpret_cast<const v8float*>(row_w));
        const v8float ue0 =
            *(reinterpret_cast<const v8float*>(row_e));
        const v8float ce0 =
            *(reinterpret_cast<const v8float*>(c_east));
        const v8float dN0 =
            fpmac(u_n0, u_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dS0 =
            fpmac(u_s0, u_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dW0 =
            fpmac(uw0, u_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dE0 =
            fpmac(ue0, u_c0, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float da0 =
            fpmac(fpmul(c0, 0, kLaneOffsets, dN0, 0, kLaneOffsets),
                  fpmul(c_s0, 0, kLaneOffsets, dS0, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float db0 =
            fpmac(fpmul(c0, 0, kLaneOffsets, dW0, 0, kLaneOffsets),
                  fpmul(ce0, 0, kLaneOffsets, dE0, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float delta0 =
            fpmac(da0, db0, 0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float out0 =
            fpmac(u_c0,
                  fpmul(update_scale, 0, kLaneOffsets, delta0, 0,
                        kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        *(reinterpret_cast<v8float*>(out_base + base_c)) = out0;

        const v8float u_n1 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_n + 8));
        const v8float u_s1 =
            *(reinterpret_cast<const v8float*>(j_update_base + base_s + 8));

        for (int lane = 0; lane < srad_cfg::kLanes; ++lane)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kLanes, srad_cfg::kLanes) {
                const int col = 8 + lane;
                const int west_col = col - 1;
                const int east_col =
                    (col == BLOCK_COL - 1) ? BLOCK_COL - 1 : col + 1;
                row_w[lane] = row_j[west_col];
                row_e[lane] = row_j[east_col];
                c_east[lane] = c_row[east_col];
            }

        const v8float uw1 =
            *(reinterpret_cast<const v8float*>(row_w));
        const v8float ue1 =
            *(reinterpret_cast<const v8float*>(row_e));
        const v8float ce1 =
            *(reinterpret_cast<const v8float*>(c_east));
        const v8float dN1 =
            fpmac(u_n1, u_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dS1 =
            fpmac(u_s1, u_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dW1 =
            fpmac(uw1, u_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float dE1 =
            fpmac(ue1, u_c1, 0, kLaneOffsets, neg_one, 0, kLaneOffsets);
        const v8float da1 =
            fpmac(fpmul(c1, 0, kLaneOffsets, dN1, 0, kLaneOffsets),
                  fpmul(c_s1, 0, kLaneOffsets, dS1, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float db1 =
            fpmac(fpmul(c1, 0, kLaneOffsets, dW1, 0, kLaneOffsets),
                  fpmul(ce1, 0, kLaneOffsets, dE1, 0, kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float delta1 =
            fpmac(da1, db1, 0, kLaneOffsets, one, 0, kLaneOffsets);
        const v8float out1 =
            fpmac(u_c1,
                  fpmul(update_scale, 0, kLaneOffsets, delta1, 0,
                        kLaneOffsets),
                  0, kLaneOffsets, one, 0, kLaneOffsets);
        *(reinterpret_cast<v8float*>(out_base + base_c + 8)) = out1;

        *(reinterpret_cast<v8float*>(c_south_row)) = c0;
        *(reinterpret_cast<v8float*>(c_south_row + 8)) = c1;
    }
}
