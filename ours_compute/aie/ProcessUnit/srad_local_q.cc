#include <adf.h>
#include <aie_api/aie.hpp>

#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

// Ours-compute K1: local SRAD directional-difference computation.
//
// DAG nodes:
//   dN/dS/dW/dE
// This variant intentionally does not precompute the algebraically reduced
// N term. K2 recomputes the original SRAD coefficient expression with the
// same division structure as the FPGA-style compute path.
//
// K1 per-pixel approximate arithmetic:
//   add/sub: 4
//   mul: 0
//   div: 0
//
// Each graph firing handles one packed 8x8 block.
//
// Mid local buffer layout per pixel, in record order:
//   unused, dN, dS, dW, dE, JC
//
// No K1 output is visible to GMIO/DDR.

namespace {

inline int block_index(int r, int c) {
    return r * BLOCK_COL + c;
}

inline int block_north_row(int r) {
    return (r == 0) ? 0 : r - 1;
}

inline int block_south_row(int r) {
    return (r == BLOCK_ROW - 1) ? BLOCK_ROW - 1 : r + 1;
}

inline int block_west_col(int c) {
    return (c == 0) ? 0 : c - 1;
}

inline int block_east_col(int c) {
    return (c == BLOCK_COL - 1) ? BLOCK_COL - 1 : c + 1;
}

} // namespace

extern "C" void srad_local_q(srad_image_input_buffer& in_j,
                             output_buffer<float>& mid_buffer) {
    const float* __restrict j_base = in_j.data();
    float* __restrict mid_base = mid_buffer.data();
    int mid = 0;

    for (int i = BLOCK_ROW - 1; i >= 0; --i) {
        const int iN = block_north_row(i);
        const int iS = block_south_row(i);

        for (int j = BLOCK_COL - 1; j >= 0; --j)
            chess_prepare_for_pipelining
            chess_loop_range(BLOCK_COL, BLOCK_COL) {
                const int jW = block_west_col(j);
                const int jE = block_east_col(j);
                const int idx = block_index(i, j);

                const float JC = j_base[idx];
                const float JN = j_base[block_index(iN, j)];
                const float JS = j_base[block_index(iS, j)];
                const float JW = j_base[block_index(i, jW)];
                const float JE = j_base[block_index(i, jE)];

                const float dN = JN - JC;
                const float dS = JS - JC;
                const float dW = JW - JC;
                const float dE = JE - JC;
                mid_base[mid++] = srad_math::kZero;
                mid_base[mid++] = dN;
                mid_base[mid++] = dS;
                mid_base[mid++] = dW;
                mid_base[mid++] = dE;
                mid_base[mid++] = JC;
            }
    }
}
