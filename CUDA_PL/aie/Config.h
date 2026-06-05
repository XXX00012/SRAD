#pragma once

namespace srad_cfg {

constexpr int kTileCol = 7;
constexpr int kPrepareTileRow = 1;
constexpr int kReduceTileRow = 2;
constexpr int kCoeffTileRow = 3;
constexpr int kUpdateTileRow = 4;

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kPixels = kRows * kCols;
constexpr int kLanes = 8;
constexpr int kVectorLoopsPerImage = kPixels / kLanes;
constexpr int kOpenClReductionThreads = 256;

constexpr int kBlockRows = 16;
constexpr int kBlockCols = 16;
constexpr int kBlockPixels = kBlockRows * kBlockCols;
constexpr int kTileRows = (kRows + kBlockRows - 1) / kBlockRows;
constexpr int kTileCols = (kCols + kBlockCols - 1) / kBlockCols;
constexpr int kTileCount = kTileRows * kTileCols;

constexpr int kCoeffInputRows = kRows;
constexpr int kCoeffInputCols = kCols;
constexpr int kCoeffInputElems = kPixels;

constexpr int kUpdateCInputRows = kRows;
constexpr int kUpdateCInputCols = kCols;
constexpr int kUpdateCInputElems = kPixels;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kBlockBytes = kBlockPixels * kScalarBytes;
constexpr int kCoeffInputBytes = kCoeffInputElems * kScalarBytes;
constexpr int kUpdateCInputBytes = kUpdateCInputElems * kScalarBytes;
constexpr int kScalarPacketElems = kLanes;
constexpr int kScalarPacketBytes = kScalarPacketElems * kScalarBytes;

constexpr int kDefaultIterations = 1;

}
