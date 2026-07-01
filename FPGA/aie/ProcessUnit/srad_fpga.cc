#include <adf.h>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// FPGA-style single-core SRAD baseline. Each graph iteration is one full SRAD
// iteration: the AIE first consumes the whole image to compute q0sqr, then
// consumes tiled halo data and emits one updated value per image cell.

namespace {

void process_update_tile(int tile,
                         float q0sqr,
                         float lambda,
                         srad_float_input_stream* in_compute,
                         srad_float_output_stream* out_index,
                         srad_float_output_stream* out_value);

inline int image_index(int row, int col) {
    return row + srad_cfg::kRows * col;
}

inline int tile_start_row(int tile) {
    return (tile / srad_cfg::kColTiles) * srad_cfg::kTileRows;
}

inline int tile_start_col(int tile) {
    return (tile % srad_cfg::kColTiles) * srad_cfg::kTileCols;
}

inline float compute_c_loc(float cC,
                           float dN,
                           float dS,
                           float dW,
                           float dE,
                           float q0sqr) {
    if (srad_cfg::kBypassCoeffMath) {
        return 1.0f;
    }

    const float G2 =
        (dN * dN + dS * dS + dW * dW + dE * dE) / (cC * cC);
    const float L = (dN + dS + dW + dE) / cC;
    const float num = (G2 / 2.0f) - ((L * L) / 16.0f);
    const float denl = 1.0f + (L / 4.0f);
    const float qsqr = num / (denl * denl);
    const float denq = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    const float c_loc = 1.0f / (1.0f + denq);

    return srad_math::clamp01(c_loc);
}

inline float compute_q0sqr_from_sums(float sum, float sum2) {
    const float pixels = static_cast<float>(srad_cfg::kPixels);
    const float mean = sum / pixels;
    const float variance = (sum2 / pixels) - (mean * mean);
    return (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;
}

inline void read_row(srad_float_input_stream* in_compute,
                     float row[srad_cfg::kInputTileCols]) {
    for (int col = 0; col < srad_cfg::kInputTileCols; ++col)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kInputTileCols, srad_cfg::kInputTileCols) {
            row[col] = readincr(in_compute);
        }
}

inline void compute_coeff_row(const float north[srad_cfg::kInputTileCols],
                              const float center[srad_cfg::kInputTileCols],
                              const float south[srad_cfg::kInputTileCols],
                              int global_row,
                              int tile_start_col,
                              float q0sqr,
                              float coeff[srad_cfg::kTileCols + 1]) {
    for (int local_col = 0; local_col < srad_cfg::kTileCols + 1; ++local_col)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kTileCols + 1, srad_cfg::kTileCols + 1) {
            const int center_col = local_col + srad_cfg::kTileLeftHaloCols;
            const int global_col = tile_start_col + local_col;
            const int west_col = center_col - 1;
            const int east_col = center_col + 1;
            const float cC = center[center_col];
            const float dN =
                (global_row == 0) ? 0.0f : north[center_col] - cC;
            const float dS =
                (global_row == srad_cfg::kRows - 1) ? 0.0f : south[center_col] - cC;
            const float dW =
                (global_col == 0) ? 0.0f : center[west_col] - cC;
            const float dE =
                (global_col == srad_cfg::kCols - 1) ? 0.0f : center[east_col] - cC;
            coeff[local_col] = compute_c_loc(cC, dN, dS, dW, dE, q0sqr);
        }
}

inline void emit_row(const float north[srad_cfg::kInputTileCols],
                     const float center[srad_cfg::kInputTileCols],
                     const float south[srad_cfg::kInputTileCols],
                     const float coeff_center[srad_cfg::kTileCols + 1],
                     const float coeff_south[srad_cfg::kTileCols + 1],
                     int global_row,
                     int tile_start_col,
                     float lambda,
                     srad_float_output_stream* out_index,
                     srad_float_output_stream* out_value) {
    for (int local_col = 0; local_col < srad_cfg::kTileCols; ++local_col)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kTileCols, srad_cfg::kTileCols) {
            const int center_col = local_col + srad_cfg::kTileLeftHaloCols;
            const int west_col = center_col - 1;
            const int east_col = center_col + 1;
            const int global_col = tile_start_col + local_col;
            const bool valid_cell =
                (global_row < srad_cfg::kRows) && (global_col < srad_cfg::kCols);
            const float cC = center[center_col];
            const float dN =
                valid_cell && global_row != 0 ? north[center_col] - cC : 0.0f;
            const float dS =
                valid_cell && global_row != srad_cfg::kRows - 1
                    ? south[center_col] - cC
                    : 0.0f;
            const float dW =
                valid_cell && global_col != 0 ? center[west_col] - cC : 0.0f;
            const float dE =
                valid_cell && global_col != srad_cfg::kCols - 1
                    ? center[east_col] - cC
                    : 0.0f;
            const float cN = valid_cell ? coeff_center[local_col] : 0.0f;
            const float cW = cN;
            const float cS =
                valid_cell && global_row != srad_cfg::kRows - 1
                    ? coeff_south[local_col]
                    : cN;
            const float cE =
                valid_cell && global_col != srad_cfg::kCols - 1
                    ? coeff_center[local_col + 1]
                    : cN;
            const float D = cN * dN + cS * dS + cW * dW + cE * dE;
            const float value = cC + (lambda * D) / 4.0f;
            const int out_idx =
                valid_cell ? image_index(global_row, global_col) : -1;

            writeincr(out_index, static_cast<float>(out_idx));
            writeincr(out_value, valid_cell ? value : 0.0f);
        }
}

void process_update_tile(int tile,
                         float q0sqr,
                         float lambda,
                         srad_float_input_stream* in_compute,
                         srad_float_output_stream* out_index,
                         srad_float_output_stream* out_value) {
    const int start_row = tile_start_row(tile);
    const int start_col = tile_start_col(tile);

    alignas(32) float row0[srad_cfg::kInputTileCols];
    alignas(32) float row1[srad_cfg::kInputTileCols];
    alignas(32) float row2[srad_cfg::kInputTileCols];
    alignas(32) float row3[srad_cfg::kInputTileCols];
    alignas(32) float coeff0[srad_cfg::kTileCols + 1];
    alignas(32) float coeff1[srad_cfg::kTileCols + 1];

    read_row(in_compute, row0);
    read_row(in_compute, row1);
    read_row(in_compute, row2);

    compute_coeff_row(row0, row1, row2, start_row, start_col, q0sqr, coeff0);

    for (int local_row = 0; local_row < srad_cfg::kTileRows; ++local_row) {
        read_row(in_compute, row3);

        const int global_row = start_row + local_row;
        const int south_global_row =
            (global_row >= srad_cfg::kRows - 1)
                ? srad_cfg::kRows - 1
                : global_row + 1;
        compute_coeff_row(row1,
                          row2,
                          row3,
                          south_global_row,
                          start_col,
                          q0sqr,
                          coeff1);
        emit_row(row0,
                 row1,
                 row2,
                 coeff0,
                 coeff1,
                 global_row,
                 start_col,
                 lambda,
                 out_index,
                 out_value);

        for (int col = 0; col < srad_cfg::kInputTileCols; ++col)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kInputTileCols, srad_cfg::kInputTileCols) {
                row0[col] = row1[col];
                row1[col] = row2[col];
                row2[col] = row3[col];
            }

        for (int col = 0; col < srad_cfg::kTileCols + 1; ++col)
            chess_prepare_for_pipelining
            chess_loop_range(srad_cfg::kTileCols + 1, srad_cfg::kTileCols + 1) {
                coeff0[col] = coeff1[col];
        }
    }
}

} // namespace

void srad_fpga_v5(srad_float_input_stream* in_params,
                  srad_float_input_stream* in_compute,
                  srad_float_output_stream* out_index,
                  srad_float_output_stream* out_value) {
    const float lambda = readincr(in_params);
    const float unused_param = readincr(in_params);
    (void)unused_param;

    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < srad_cfg::kPixels; ++i)
        chess_prepare_for_pipelining
        {
            const float v = readincr(in_compute);
            sum += v;
            sum2 += v * v;
        }
    const float q0sqr = compute_q0sqr_from_sums(sum, sum2);

    for (int tile = 0; tile < srad_cfg::kTilesPerIteration; ++tile) {
        process_update_tile(tile, q0sqr, lambda, in_compute, out_index, out_value);
    }
}
