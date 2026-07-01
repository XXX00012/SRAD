#pragma once

namespace srad_cfg {

// This version is deliberately single-lane and row-stream based.
constexpr int kParallelLanes = 1;
constexpr int kTopPlWorkers = 1;

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
constexpr int kStatsInputSampleElems = kRowPhysElems;
constexpr int kStatsOutputElems = 2;
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
constexpr int kBoardGraphRowsPerIteration = kBoardStrips * kBoardRowsPerStrip;
constexpr int kBoardGraphRows = kBoardGraphRowsPerIteration * kBoardIterations;
constexpr int kBoardImageBytes = kBoardPixels * kScalarBytes;
constexpr int kBoardOutputBytes = kBoardPixels * kScalarBytes;


constexpr int kInputObjectFifoDepth = 2;
constexpr int kDelayedInputObjectFifoDepth = 2;
constexpr int kMidObjectFifoDepth = 2;
constexpr int kOutputObjectFifoDepth = 2;

constexpr int kAieTileCol = 1;
constexpr int kStatsTileRow = 2;
constexpr int kLocalQTileRow = 0;
constexpr int kCoeffUpdateTileRow = 1;


}  // namespace srad_cfg
