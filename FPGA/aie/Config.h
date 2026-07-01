#pragma once

namespace srad_cfg {

constexpr int kParallelLanes = 1;
constexpr int kTopPlWorkers = 1;

constexpr int kTileCol = 7;
constexpr int kFusedTileRow = 2;

constexpr int kRows = 4000;
constexpr int kCols = 4000;
constexpr int kPixels = kRows * kCols;

constexpr int kLanes = 8;
constexpr int kTileRows = 16;
constexpr int kTileCols = 16;
constexpr int kTileLeftHaloCols = 1;
constexpr int kInputTileRows = kTileRows + 3;
constexpr int kInputTileCols = 24;
constexpr int kActiveInputTileCols = kTileLeftHaloCols + kTileCols + 2;
constexpr int kRowTiles = (kRows + kTileRows - 1) / kTileRows;
constexpr int kColTiles = (kCols + kTileCols - 1) / kTileCols;
constexpr int kTilesPerIteration = kRowTiles * kColTiles;
constexpr int kTileInputElems = kInputTileRows * kInputTileCols;
constexpr int kTileOutputElems = kTileRows * kTileCols;

constexpr int kParamElems = 2;
constexpr int kComputeStreamElems =
    kPixels + kTilesPerIteration * kTileInputElems;
constexpr int kOutputStreamElems = kTilesPerIteration * kTileOutputElems;
constexpr int kOutputElems = kPixels;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kOutputBytes = kOutputElems * kScalarBytes;
constexpr int kScalarPacketElems = kParamElems;
constexpr int kScalarPacketBytes = kScalarPacketElems * kScalarBytes;

constexpr int kDefaultIterations = 100;

static_assert((kRows % kTileRows) == 0,
              "board rows should be an exact multiple of tile rows");
static_assert((kCols % kTileCols) == 0,
              "board cols should be an exact multiple of tile cols");
static_assert(kInputTileCols >= kActiveInputTileCols,
              "input tile row must cover left halo, tile cols, south/east coeff lookahead");

}
