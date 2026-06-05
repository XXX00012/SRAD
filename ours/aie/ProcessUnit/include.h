#pragma once

#include "../Config.h"

using srad_data_t = float;

#define GRIDROW srad_cfg::kRows
#define GRIDCOL srad_cfg::kCols
#define GRIDDEPTH 1
#define TOTAL_INPUT (GRIDROW * GRIDCOL * GRIDDEPTH)

#define ROW srad_cfg::kInputLogicalRows
#define COL srad_cfg::kInputRowElems
#define OUT_COL srad_cfg::kOutputRowElems
#define ROW_ELEMS srad_cfg::kInputRowElems
#define TILE_SIZE srad_cfg::kInputRowElems

#define NBYTES srad_cfg::kScalarBytes
#define INPUT_FILE "./data/plio_ours_j.txt"
#define OUTPUT_FILE "./data/TestOutputS.txt"

namespace srad_math {

constexpr float kZero = 0.0f;
constexpr float kOne = 1.0f;
constexpr float kQuarter = 0.25f;
constexpr float kHalf = 0.5f;
constexpr float kOneSixteenth = 1.0f / 16.0f;

inline float clamp01(float v) {
    if (v < kZero) return kZero;
    if (v > kOne) return kOne;
    return v;
}

inline int north_row(int r) {
    return (r == 0) ? 0 : r - 1;
}

inline int south_row(int r) {
    return (r == srad_cfg::kRows - 1) ? srad_cfg::kRows - 1 : r + 1;
}

inline int west_col(int c) {
    return (c == 0) ? 0 : c - 1;
}

inline int east_col(int c) {
    return (c == srad_cfg::kCols - 1) ? srad_cfg::kCols - 1 : c + 1;
}

inline int image_index(int r, int c) {
    return r * srad_cfg::kCols + c;
}

} // namespace srad_math
