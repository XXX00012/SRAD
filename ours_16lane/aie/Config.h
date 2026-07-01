#pragma once

namespace srad_cfg {

// 16 row-stream lanes. Each lane uses two AIE cores: local_q then coeff_update.
// Lanes are placed on AIE columns 1..4, four lanes per column, leaving column 0
// unused.
constexpr int kParallelLanes = 16;
constexpr int kTopPlWorkers = 1;
constexpr int kKernelsPerParallelLane = 2;
constexpr int kParallelLanesPerCol = 4;
constexpr int kTotalAieCores = kParallelLanes * kKernelsPerParallelLane;

constexpr int kRows = 125;
constexpr int kCols = 125;
constexpr int kPixels = kRows * kCols;
constexpr int kSimRows = 6;

constexpr int kBoardRows = 4000;
constexpr int kBoardCols = 4000;
constexpr int kBoardPixels = kBoardRows * kBoardCols;
constexpr int kBoardIterations = 100;
constexpr int kBoardRowsPerLaneMax = kBoardRows;
constexpr int kBoardLanePreContextRows = 1;

constexpr int kLanes = 8;
constexpr int kRowDataElems = kCols;
constexpr int kRowPadElems = 3;
constexpr int kRowPhysElems = kRowDataElems + kRowPadElems;

constexpr int kQ0PadIndex = 125;
constexpr int kStatSumPadIndex = 125;
constexpr int kStatSum2PadIndex = 126;

constexpr int kRowsPerCall = 1;
constexpr int kLocalQWindowRows = 4;
constexpr int kLocalQMarginRows = kLocalQWindowRows - kRowsPerCall;
constexpr int kUpdateWindowRows = 4;
constexpr int kUpdateMarginRows = kUpdateWindowRows - kRowsPerCall;
constexpr int kLocalQMarginElems = kLocalQMarginRows * kRowPhysElems;
constexpr int kUpdateMarginElems = kUpdateMarginRows * kRowPhysElems;
constexpr int kLocalQInputSampleElems = kRowsPerCall * kRowPhysElems;
constexpr int kUpdateInputSampleElems = kRowsPerCall * kRowPhysElems;
constexpr int kCenterRowLag = 2;
constexpr int kFlushRows = kCenterRowLag;

constexpr int kSradIterations = 1;
constexpr int kSimIterations = kSradIterations;
constexpr int kDefaultIterations = kBoardIterations;
constexpr int kSimInvalidRows = kCenterRowLag;

constexpr int kMidRecordElems = 4;
constexpr int kMidPlaneStride = kRowPhysElems;
constexpr int kMidElemsPerRow = kMidRecordElems * kMidPlaneStride;

constexpr int kOutputRowPhysElems = kRowPhysElems;
constexpr int kOutputElems = kPixels;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kOutputBytes = kOutputElems * kScalarBytes;
constexpr int kRowPhysBytes = kRowPhysElems * kScalarBytes;
constexpr int kMidBytesPerRow = kMidElemsPerRow * kScalarBytes;

static_assert((kMidBytesPerRow % 16) == 0,
              "K1->K2 buffer size must be 16-byte aligned");

constexpr int kRowsPerIterAll = kRows + kFlushRows;
constexpr int kRowsPerIterSim = kSimRows + kFlushRows;
constexpr int kGraphRowsAll = kRowsPerIterAll * kSradIterations;
constexpr int kGraphRowsSim = kRowsPerIterSim * kSimIterations;

static_assert(kBoardCols % kRowDataElems == 0,
              "board width must be a multiple of the AIE strip width");
constexpr int kBoardStrips = kBoardCols / kRowDataElems;
static_assert((kBoardStrips % kParallelLanes) == 0,
              "board strips must divide evenly across AIE lanes");
constexpr int kBoardStripBatches = kBoardStrips / kParallelLanes;
constexpr int kBoardRowsPerLaneStream =
    kBoardRowsPerLaneMax + kBoardLanePreContextRows + kFlushRows;
constexpr int kBoardRowsPerStrip = kBoardRowsPerLaneStream;
constexpr int kBoardGraphRowsPerIteration =
    kBoardStripBatches * kBoardRowsPerLaneStream;
constexpr int kBoardGraphRows = kBoardGraphRowsPerIteration * kBoardIterations;
constexpr int kBoardImageBytes = kBoardPixels * kScalarBytes;
constexpr int kBoardOutputBytes = kBoardPixels * kScalarBytes;


// Extra slack is needed because the PLIO input fans out to K1 and K2 while K2
// also waits for the K1 coefficient stream.
constexpr int kInputObjectFifoDepth = 4;
constexpr int kDelayedInputObjectFifoDepth = 4;
constexpr int kMidObjectFifoDepth = 4;
constexpr int kOutputObjectFifoDepth = 4;

constexpr int kAieBaseTileCol = 1;
constexpr int kAieLaneCols = 4;

constexpr int lane_tile_col(int lane) {
    return kAieBaseTileCol + lane / kParallelLanesPerCol;
}

constexpr int local_q_tile_row(int lane) {
    return (lane % kParallelLanesPerCol) * kKernelsPerParallelLane;
}

constexpr int coeff_update_tile_row(int lane) {
    return local_q_tile_row(lane) + 1;
}

static_assert(kParallelLanes == 16,
              "ours_16lane expects exactly 16 AIE row-stream lanes");
static_assert(kTotalAieCores == 32,
              "ours_16lane expects exactly 32 AIE kernels");
static_assert(kAieLaneCols * kParallelLanesPerCol == kParallelLanes,
              "16 lanes should occupy four AIE columns");
static_assert((kAieBaseTileCol == 1),
              "AIE column 0 is intentionally left empty");


}  // namespace srad_cfg
