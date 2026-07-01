#pragma once

namespace srad_cfg {

// 96 row-stream lanes. Each lane uses two AIE cores: local_q then coeff_update.
// Six TopPL CUs are arranged as 3 row blocks x 2 column workers. Each CU drives
// 16 lanes over one contiguous 2000-column half image and 1334 center rows.
// Lanes are placed on AIE columns 1..24, four lanes per column, leaving column 0
// unused.
constexpr int kTopPlColumnWorkers = 2;
constexpr int kBoardRowBlocks = 3;
constexpr int kTopPlWorkers = kTopPlColumnWorkers * kBoardRowBlocks;
constexpr int kLanesPerTopPl = 16;
constexpr int kParallelLanes = kTopPlWorkers * kLanesPerTopPl;
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
constexpr int kRowsPerRowBlock = 1334;
constexpr int kBoardPaddedRows = kRowsPerRowBlock * kBoardRowBlocks;
constexpr int kBoardIterations = 100;
constexpr int kBoardRowsPerLaneMax = kRowsPerRowBlock;
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
static_assert((kBoardStrips % kTopPlColumnWorkers) == 0,
              "board strips must divide evenly across column workers");
constexpr int kBoardStripsPerTopPl = kBoardStrips / kTopPlColumnWorkers;
static_assert(kBoardStripsPerTopPl == kLanesPerTopPl,
              "each TopPL should own one contiguous 16-strip half image");
constexpr int kWorkerCols = kLanesPerTopPl * kRowDataElems;
static_assert(kWorkerCols * kTopPlColumnWorkers == kBoardCols,
              "TopPL column worker ranges must exactly cover the image width");
static_assert(kRowsPerRowBlock * kBoardRowBlocks == kBoardPaddedRows,
              "row blocks must exactly cover the padded board height");
static_assert(kBoardPaddedRows >= kBoardRows,
              "padded board rows must cover all real board rows");
constexpr int kBoardLanesPerRowBlock = kTopPlColumnWorkers * kLanesPerTopPl;
static_assert(kBoardStrips == kBoardLanesPerRowBlock,
              "one 32-lane row block should cover all board strips");
constexpr int kBoardStripBatches = kBoardStrips / kBoardLanesPerRowBlock;
constexpr int kBoardRowsPerLaneStream =
    kBoardRowsPerLaneMax + kBoardLanePreContextRows + kFlushRows;
constexpr int kBoardRowsPerStrip = kBoardRowsPerLaneStream;
constexpr int kBoardGraphRowsPerIteration =
    kBoardRowsPerLaneStream;
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
constexpr int kAieLaneCols = 24;

constexpr int lane_tile_col(int lane) {
    return kAieBaseTileCol + lane / kParallelLanesPerCol;
}

constexpr int local_q_tile_row(int lane) {
    return (lane % kParallelLanesPerCol) * kKernelsPerParallelLane;
}

constexpr int coeff_update_tile_row(int lane) {
    return local_q_tile_row(lane) + 1;
}

static_assert(kParallelLanes == 96,
              "ours_96lane expects exactly 96 AIE row-stream lanes");
static_assert(kTopPlWorkers == 6,
              "ours_96lane expects six TopPL workers");
static_assert(kTopPlColumnWorkers == 2,
              "ours_96lane expects two TopPL column workers per row block");
static_assert(kBoardRowBlocks == 3,
              "ours_96lane expects three padded row blocks");
static_assert(kLanesPerTopPl == 16,
              "each TopPL worker should drive 16 AIE row-stream lanes");
static_assert(kTotalAieCores == 192,
              "ours_96lane expects exactly 192 AIE kernels");
static_assert(kAieLaneCols * kParallelLanesPerCol == kParallelLanes,
              "96 lanes should occupy twenty-four AIE columns");
static_assert((kAieBaseTileCol == 1),
              "AIE column 0 is intentionally left empty");


}  // namespace srad_cfg
