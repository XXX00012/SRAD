#include <adf.h>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// OpenCL-v0 faithful float32 SRAD update stage:
//   d_I[ei] = d_I[ei] + 0.25f * lambda * D
// All inputs are materialized buffers, matching the source global-memory
// kernel boundary.

void srad2_kernel(srad_image_input_buffer& in_j,
                  srad_image_input_buffer& in_c,
                  srad_image_input_buffer& in_dN,
                  srad_image_input_buffer& in_dS,
                  srad_image_input_buffer& in_dW,
                  srad_image_input_buffer& in_dE,
                  srad_scalar_input_buffer& in_lambda,
                  output_buffer<float>& out_j_next) {
    const float* __restrict j_base = in_j.data();
    const float* __restrict c_base = in_c.data();
    const float* __restrict dN_base = in_dN.data();
    const float* __restrict dS_base = in_dS.data();
    const float* __restrict dW_base = in_dW.data();
    const float* __restrict dE_base = in_dE.data();
    const float* __restrict lambda_base = in_lambda.data();
    float* __restrict out_base = out_j_next.data();

    const float lambda = lambda_base[0];
    static int block_id = 0;
    const int bx = block_id;
    block_id = (block_id + 1) % srad_cfg::kTileCount;

    const int tiles_per_col = srad_cfg::kTileRows;
    const int tile_col = bx / tiles_per_col;
    const int tile_row = bx - tile_col * tiles_per_col;
    const int row0 = tile_row * srad_cfg::kBlockRows;
    const int col0 = tile_col * srad_cfg::kBlockCols;

    for (int idx = 0; idx < srad_cfg::kBlockPixels; ++idx)
        chess_prepare_for_pipelining
        chess_loop_range(srad_cfg::kBlockPixels, srad_cfg::kBlockPixels) {
            const int i = idx % srad_cfg::kBlockRows;
            const int j = idx / srad_cfg::kBlockRows;
            const int row = row0 + i;
            const int col = col0 + j;
            const int south_row =
                (row >= srad_cfg::kRows - 1) ? srad_cfg::kRows - 1 : row + 1;
            const int east_col =
                (col >= srad_cfg::kCols - 1) ? srad_cfg::kCols - 1 : col + 1;
            const int c_idx = row + srad_cfg::kRows * col;
            const int c_s_idx = south_row + srad_cfg::kRows * col;
            const int c_e_idx = row + srad_cfg::kRows * east_col;

            const float cN = c_base[c_idx];
            const float cW = c_base[c_idx];
            const float cS = c_base[c_s_idx];
            const float cE = c_base[c_e_idx];

            const float D =
                cN * dN_base[idx] +
                cS * dS_base[idx] +
                cW * dW_base[idx] +
                cE * dE_base[idx];

            out_base[idx] = j_base[idx] + 0.25f * lambda * D;
        }
}
