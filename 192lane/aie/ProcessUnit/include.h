#pragma once

#include "../Config.h"

using srad_data_t = float;

#define GRIDROW srad_cfg::kRows
#define GRIDCOL srad_cfg::kCols
#define GRIDDEPTH 1
#define TOTAL_INPUT (GRIDROW * GRIDCOL * GRIDDEPTH)

#define ROW_PHYS srad_cfg::kRowPhysElems
#define ROW_DATA srad_cfg::kRowDataElems
#define COL srad_cfg::kRowPhysElems
#define NBYTES srad_cfg::kScalarBytes
#define INPUT_FILE "./data/plio_ours_j_0.txt"
#define OUTPUT_FILE "./data/aiesim_j_next_merged_0.txt"

namespace srad_math {

constexpr float kZero = 0.0f;
constexpr float kOne = 1.0f;
constexpr float kQuarter = 0.25f;
constexpr float kHalf = 0.5f;
constexpr float kOneSixteenth = 1.0f / 16.0f;

} // namespace srad_math
