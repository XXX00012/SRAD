#pragma once

namespace srad_cfg {

constexpr int kTileCol = 7;
constexpr int kFusedTileRow = 2;

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kPixels = kRows * kCols;
constexpr int kLanes = 8;
constexpr int kVectorLoopsPerImage = kPixels / kLanes;

constexpr float kLambdaDefault = 0.5f;
constexpr bool kBypassCoeffMath = false;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kScalarPacketElems = kLanes;
constexpr int kScalarPacketBytes = kScalarPacketElems * kScalarBytes;

constexpr int kDefaultIterations = 1;

}
