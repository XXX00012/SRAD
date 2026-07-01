#pragma once

namespace srad_cfg {

// One TopPL drives 12 independent row-stream AIE lanes.
constexpr int kParallelLanes = 12;
constexpr int kTopPlWorkers = 1;
constexpr int kAieStartCol = 1;
constexpr int kAieArrayCols = 3;
constexpr int kAieArrayRows = 8;
constexpr int kKernelsPerParallelLane = 2;
constexpr int kParallelLanesPerCol =
    kAieArrayRows / kKernelsPerParallelLane;
constexpr int kTotalAieCores =
    kParallelLanes * kKernelsPerParallelLane;

constexpr int kRows = 125;
constexpr int kCols = 125;
constexpr int kPixels = kRows * kCols;
constexpr int kSimRows = 6;

constexpr int kBoardRows = 4000;
constexpr int kBoardCols = 4000;
constexpr int kBoardPixels = kBoardRows * kBoardCols;
constexpr int kBoardIterations = 100;

constexpr int kLanes = 8;
constexpr int kRowDataElems = kCols;
constexpr int kRowPadElems = 3;
constexpr int kRowPhysElems = 128;
constexpr int kRowChunks =  (kRowDataElems + kLanes - 1) / kLanes;

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

constexpr int kMidRecordElems = 6;
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
constexpr int kBoardRowsPerStrip = kBoardRows + kFlushRows;
constexpr int kBoardStripGroups =
    (kBoardStrips + kParallelLanes - 1) / kParallelLanes;
constexpr int kBoardGraphRowsPerIteration =
    kBoardStripGroups * kBoardRowsPerStrip;
constexpr int kBoardGraphRows = kBoardGraphRowsPerIteration * kBoardIterations;
constexpr int kBoardImageBytes = kBoardPixels * kScalarBytes;
constexpr int kBoardOutputBytes = kBoardPixels * kScalarBytes;


constexpr int kInputObjectFifoDepth = 2;
constexpr int kDelayedInputObjectFifoDepth = 2;
constexpr int kMidObjectFifoDepth = 2;
constexpr int kOutputObjectFifoDepth = 2;

static_assert(kTopPlWorkers == 1,
              "ours_12lane board path expects one TopPL worker");
static_assert(kParallelLanes == 12,
              "ours_12lane board path expects 12 AIE lanes");
static_assert(kAieStartCol == 1,
              "ours_12lane keeps AIE column 0 empty");
static_assert(kParallelLanesPerCol == 4,
              "8 AIE rows with two kernels per lane gives four lanes/col");
static_assert(kAieArrayCols * kParallelLanesPerCol >= kParallelLanes,
              "placement columns must cover all AIE lanes");
static_assert(kTotalAieCores <= kAieArrayCols * kAieArrayRows,
              "placement grid must cover all AIE kernels");


}  // namespace srad_cfg
