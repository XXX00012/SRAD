#pragma once

namespace srad_cfg {

constexpr int kTileCol = 7;
constexpr int kLocalQTileRow = 2;
constexpr int kCoeffUpdateTileRow = 3;

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kPixels = kRows * kCols;
constexpr int kLanes = 8;
constexpr int kVectorLoopsPerImage = kPixels / kLanes;

// Ours-undivide pressure experiment: 32x32 image processed as 16x16 fused tiles.
constexpr int kBlockRows = 16;
constexpr int kBlockCols = 16;
constexpr int kBlockPixels = kBlockRows * kBlockCols;
constexpr int kBlocksPerRow = kCols / kBlockCols;
constexpr int kBlocksPerCol = kRows / kBlockRows;
constexpr int kBlockCount = kBlocksPerRow * kBlocksPerCol;
constexpr int kVectorLoopsPerBlock = kBlockPixels / kLanes;

constexpr int kMidRecordElems = 6;
constexpr int kMidElems = kPixels * kMidRecordElems;
constexpr int kMidElemsPerBlock = kBlockPixels * kMidRecordElems;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;
constexpr int kPressureRowsPerGroup = kBlockRows;
constexpr int kPressureVectorsPerGroup = kBlockRows * (kBlockCols / kLanes);

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kBlockBytes = kBlockPixels * kScalarBytes;
constexpr int kMidBytes = kMidElems * kScalarBytes;
constexpr int kMidBytesPerBlock = kMidElemsPerBlock * kScalarBytes;
constexpr int kScalarPacketElems = kLanes;
constexpr int kScalarPacketBytes = kScalarPacketElems * kScalarBytes;
constexpr int kScalarPacketsPerImageBytes = kBlockCount * kScalarPacketBytes;

constexpr int kDefaultIterations = 1;

}
