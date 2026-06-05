#pragma once

namespace srad_cfg {

constexpr int kAieArrayCols = 50;
constexpr int kAieArrayRows = 8;
constexpr int kParallelLanes = 200;
constexpr int kInputLanesPerPlio = 1;
constexpr int kInputPlioGroups =
    (kParallelLanes + kInputLanesPerPlio - 1) / kInputLanesPerPlio;
constexpr int kOutputLanesPerPlio = 2;
constexpr int kOutputPlioGroups =
    (kParallelLanes + kOutputLanesPerPlio - 1) / kOutputLanesPerPlio;
constexpr int kTopPlWorkers = 4;
constexpr int kWorkerLanes = kParallelLanes / kTopPlWorkers;
constexpr int kWorkerOutputGroups = kOutputPlioGroups / kTopPlWorkers;
constexpr int kKernelsPerParallelLane = 2;
constexpr int kParallelLanesPerCol =
    kAieArrayRows / kKernelsPerParallelLane;
constexpr int kTotalAieCores =
    kParallelLanes * kKernelsPerParallelLane;

// Full DDR image shape. Change these two values when testing larger N x N
// images; the AIE tile shape below stays fixed.
constexpr int kRows = 4000;
constexpr int kCols = 4000;
constexpr int kPixels = kRows * kCols;

constexpr int kLanes = 8;
constexpr int kVectorLoopsPerImage = kPixels / kLanes;

// AIE computes one 16x16 output tile per graph firing. PL sends the full
// SRAD halo around that tile: top=1, bottom=2, left=1, right=2. The logical
// 19 columns are rounded to 24 physical columns so every PLIO transfer is
// 64-bit and 8-lane aligned.
constexpr int kOutputTileRows = 16;
constexpr int kOutputTileCols = 16;
constexpr int kHaloTopRows = 1;
constexpr int kHaloBottomRows = 2;
constexpr int kHaloLeftCols = 1;
constexpr int kHaloRightCols = 2;
constexpr int kInputLogicalRows =
    kHaloTopRows + kOutputTileRows + kHaloBottomRows;
constexpr int kInputLogicalCols =
    kHaloLeftCols + kOutputTileCols + kHaloRightCols;
constexpr int kInputPhysicalCols =
    ((kInputLogicalCols + kLanes - 1) / kLanes) * kLanes;
constexpr int kInputRowElems = kInputPhysicalCols;
constexpr int kOutputRowElems = kOutputTileCols;
constexpr int kQ0SqrTileRow = 0;
constexpr int kQ0SqrTileCol = kInputLogicalCols;
constexpr int kQ0SqrTileIndex =
    kQ0SqrTileRow * kInputRowElems + kQ0SqrTileCol;

// Backward-compatible aliases for older code paths. Here "tile" means the
// physical input tile sent to AIE, not the output tile.
constexpr int kTileRows = kInputLogicalRows;
constexpr int kTileCols = kInputPhysicalCols;
constexpr int kTilePixels = kTileRows * kTileCols;
constexpr int kRowElems = kInputRowElems;
constexpr int kVectorLoopsPerRow = kInputRowElems / kLanes;
constexpr int kInputTileElems = kInputLogicalRows * kInputRowElems;
constexpr int kOutputTilePixels = kOutputTileRows * kOutputTileCols;
constexpr int kOutputTileDataElems = kOutputTilePixels;
constexpr int kOutputStatElems = 2;
constexpr int kOutputStatSumIndex = kOutputTileDataElems;
constexpr int kOutputStatSum2Index = kOutputTileDataElems + 1;
constexpr int kOutputSampleElems = kOutputTileDataElems + kOutputStatElems;

constexpr int kImageInputWindowRows = kInputLogicalRows;
constexpr int kImageInputMarginRows = 0;
constexpr int kStencilTopRows = kHaloTopRows;
constexpr int kStencilBottomRows = kHaloBottomRows;
constexpr int kStencilLeftCols = kHaloLeftCols;
constexpr int kStencilRightCols = kHaloRightCols;
constexpr int kTileValidRowBegin = kHaloTopRows;
constexpr int kTileValidRowEnd = kHaloTopRows + kOutputTileRows - 1;
constexpr int kTileValidColBegin = kHaloLeftCols;
constexpr int kTileValidColEnd = kHaloLeftCols + kOutputTileCols - 1;
constexpr int kTileValidRows =
    kTileValidRowEnd - kTileValidRowBegin + 1;
constexpr int kTileValidCols =
    kTileValidColEnd - kTileValidColBegin + 1;
constexpr int kTileStrideRows = kOutputTileRows;
constexpr int kTileStrideCols = kOutputTileCols;
constexpr int kTileRowCount =
    (kRows + kTileStrideRows - 1) / kTileStrideRows;
constexpr int kTileColCount =
    (kCols + kTileStrideCols - 1) / kTileStrideCols;
constexpr int kTotalTileCount = kTileRowCount * kTileColCount;
constexpr int kTileCount = kTotalTileCount;
constexpr int kTilesPerParallelLane =
    (kTotalTileCount + kParallelLanes - 1) / kParallelLanes;

constexpr int kLaneInputStreamRows =
    kTilesPerParallelLane * kInputLogicalRows;
constexpr int kImageInputStreamRows =
    kParallelLanes * kLaneInputStreamRows;
constexpr int kImageInputSampleElems = kInputTileElems;
constexpr int kImageInputMarginElems =
    kImageInputMarginRows * kInputRowElems;
constexpr int kUpdateRowLagRows = 0;
constexpr int kGraphRunIterations = kTilesPerParallelLane;
constexpr int kSradIterations = 100;
constexpr int kInvalidOutputRows = 0;
constexpr int kOutputSourceRowOffset = 0;
constexpr int kOutputFirstRow = 0;
constexpr int kOutputLastRow = kRows - 1;
constexpr int kOutputFirstCol = 0;
constexpr int kOutputLastCol = kCols - 1;
constexpr int kOutputRows = kOutputLastRow - kOutputFirstRow + 1;
constexpr int kOutputCols = kOutputLastCol - kOutputFirstCol + 1;
constexpr int kCompareRows = kOutputRows;
constexpr int kOutputElems = kOutputRows * kOutputCols;
constexpr int kAieOutputTilesPerLane = kGraphRunIterations * kSradIterations;
constexpr int kAieOutputElemsPerLane =
    kAieOutputTilesPerLane * kOutputSampleElems;
constexpr int kAieOutputTiles =
    kParallelLanes * kAieOutputTilesPerLane;
constexpr int kAieOutputElems = kAieOutputTiles * kOutputSampleElems;

constexpr int kMidRecordElems = 6;
constexpr int kMidElemsPerTile = kOutputTilePixels * kMidRecordElems;
constexpr int kMidElemsPerRow = kMidElemsPerTile;
constexpr long long kMidElems =
    1LL * kAieOutputTiles * kMidElemsPerTile;
constexpr int kGradRecordElems = 4;
constexpr int kGradElemsPerTile = kOutputTilePixels * kGradRecordElems;
constexpr int kGradElemsPerRow = kGradElemsPerTile;
constexpr long long kGradElems =
    1LL * kAieOutputTiles * kGradElemsPerTile;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;
constexpr bool kDebugPassthroughTwoKernel = false;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kOutputBytes = kOutputElems * kScalarBytes;
constexpr int kInputRowBytes = kInputRowElems * kScalarBytes;
constexpr int kOutputRowBytes = kOutputRowElems * kScalarBytes;
constexpr int kInputTileBytes = kInputTileElems * kScalarBytes;
constexpr int kOutputSampleBytes = kOutputSampleElems * kScalarBytes;
constexpr int kRowBytes = kInputRowBytes;
constexpr long long kImageInputStreamBytes =
    1LL * kAieOutputTiles * kInputTileBytes;
constexpr long long kMidBytes = kMidElems * kScalarBytes;
constexpr int kMidBytesPerTile = kMidElemsPerTile * kScalarBytes;
constexpr int kMidBytesPerRow = kMidBytesPerTile;
constexpr long long kGradBytes = kGradElems * kScalarBytes;
constexpr int kGradBytesPerTile = kGradElemsPerTile * kScalarBytes;
constexpr int kGradBytesPerRow = kGradBytesPerTile;

constexpr int kInputObjectFifoDepth = 2;
constexpr int kDelayedInputObjectFifoDepth = 2;
constexpr int kMidObjectFifoDepth = 2;
constexpr int kGradObjectFifoDepth = 2;
constexpr int kOutputObjectFifoDepth = 2;

constexpr int kDefaultIterations = kSradIterations;

static_assert(kRows > 0, "SRAD image must have at least one row");
static_assert(kCols > 0, "SRAD image must have at least one column");
static_assert(kAieArrayRows % kKernelsPerParallelLane == 0,
              "Each lane occupies an integer number of AIE rows");
static_assert(kParallelLanesPerCol * kAieArrayCols >= kParallelLanes,
              "AIE array does not have enough tile slots for all lanes");
static_assert(kTotalAieCores == 400,
              "ours_400 is expected to instantiate exactly 400 kernels");
static_assert(kParallelLanes % kInputLanesPerPlio == 0,
              "Input PLIO grouping expects an even lane count");
static_assert(kInputPlioGroups == 200,
              "ours_400 should expose 200 input PLIOs");
static_assert(kParallelLanes % kOutputLanesPerPlio == 0,
              "Output PLIO grouping expects an even lane count");
static_assert(kOutputPlioGroups == 100,
              "ours_400 should expose 100 output PLIO groups");
static_assert(kTopPlWorkers == 4,
              "ours_400 split PL path expects four TopPL workers");
static_assert(kParallelLanes % kTopPlWorkers == 0,
              "TopPL workers must split input lanes evenly");
static_assert(kOutputPlioGroups % kTopPlWorkers == 0,
              "TopPL workers must split output PLIO groups evenly");
static_assert(kWorkerLanes == 50,
              "ours_400 expects 50 AIE lanes per TopPL worker");
static_assert(kWorkerOutputGroups == 25,
              "ours_400 expects 25 output PLIO groups per TopPL worker");
static_assert(kInputLogicalRows == kImageInputWindowRows,
              "One graph firing consumes one full logical input tile");
static_assert(kTileValidRows > 0, "SRAD tile has no valid output rows");
static_assert(kTileValidCols > 0, "SRAD tile has no valid output columns");
static_assert(kOutputRows > 0, "SRAD compact output needs valid rows");
static_assert(kOutputCols > 0, "SRAD compact output needs valid columns");
static_assert(kOutputSourceRowOffset >= 0,
              "SRAD output source row offset must be non-negative");
static_assert(kPixels % 2 == 0,
              "64-bit PLIO text input generation expects an even pixel count");
static_assert(kInputLogicalRows == 19,
              "This SRAD schedule expects a 19-row logical input tile");
static_assert(kInputLogicalCols == 19,
              "This SRAD schedule expects a 19-column logical input tile");
static_assert(kInputPhysicalCols >= kInputLogicalCols,
              "Physical input tile must cover the logical halo tile");
static_assert(kQ0SqrTileRow >= 0 && kQ0SqrTileRow < kInputLogicalRows,
              "q0sqr row must be inside the physical input tile");
static_assert(kQ0SqrTileCol >= kInputLogicalCols,
              "q0sqr must be stored in input tile padding, not J data");
static_assert(kQ0SqrTileCol < kInputPhysicalCols,
              "q0sqr padding slot must fit inside the physical input row");
static_assert(kInputPhysicalCols % 2 == 0,
              "64-bit PLIO requires an even physical input row");
static_assert(kInputPhysicalCols % kLanes == 0,
              "AIE row loops expect input rows to be lane-aligned");
static_assert(kImageInputSampleElems % 2 == 0,
              "64-bit PLIO requires an even input tile");
static_assert(kOutputRowElems % 2 == 0,
              "64-bit PLIO requires an even output row");
static_assert(kOutputSampleElems % 2 == 0,
              "64-bit PLIO requires an even output tile");
static_assert(kOutputStatElems == 2,
              "Ours output protocol appends tile_sum and tile_sum2");
static_assert(kOutputElems == kPixels,
              "Ping-pong iterations require compact output to cover the full image");
static_assert(kRows % kTileStrideRows == 0,
              "Tile stats q0 path expects no partial output tile rows");
static_assert(kCols % kTileStrideCols == 0,
              "Tile stats q0 path expects no partial output tile cols");
static_assert(kOutputTileCols % kLanes == 0,
              "AIE output rows are expected to be lane-aligned");

}
