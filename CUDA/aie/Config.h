#pragma once

namespace srad_cfg {

constexpr int kParallelLanes = 1;
constexpr int kPlStageCount = 4;

constexpr int kBoardRows = 4000;
constexpr int kBoardCols = 4000;
constexpr int kBoardPixels = kBoardRows * kBoardCols;
constexpr int kBoardIterations = 100;

constexpr int kRows = kBoardRows;
constexpr int kCols = kBoardCols;
constexpr int kPixels = kBoardPixels;
constexpr int kDefaultIterations = kBoardIterations;
constexpr int kSradIterations = kBoardIterations;
constexpr int kSimIterations = 1;

constexpr int kCudaBlockElems = 512;
constexpr int kLanes = 8;
constexpr int kBlockVectors = kCudaBlockElems / kLanes;
constexpr int kCudaBlocks =
    (kBoardPixels + kCudaBlockElems - 1) / kCudaBlockElems;

constexpr float kLambdaDefault = 0.5f;

constexpr int kScalarBytes = sizeof(float);
constexpr int kImageBytes = kPixels * kScalarBytes;
constexpr int kOutputElems = kPixels;
constexpr int kOutputBytes = kOutputElems * kScalarBytes;
constexpr int kBoardImageBytes = kBoardPixels * kScalarBytes;
constexpr int kBoardOutputBytes = kBoardPixels * kScalarBytes;

constexpr int kPrepareInputElems = kCudaBlockElems;
constexpr int kPrepareOutputElems = kCudaBlockElems;

constexpr int kMetaPacketElems = 8;
constexpr int kReducePairElems = 2 * kCudaBlockElems;
constexpr int kReducePacketElems = kMetaPacketElems + kReducePairElems;
constexpr int kReduceInputElems = kReducePacketElems;
constexpr int kReduceOutputElems = 2;

constexpr int kCoeffInputPlanes = 5;
constexpr int kCoeffInputElems = kCoeffInputPlanes * kCudaBlockElems;
constexpr int kCoeffOutputElems = kCudaBlockElems;

constexpr int kUpdateInputPlanes = 8;
constexpr int kUpdateInputElems = kUpdateInputPlanes * kCudaBlockElems;
constexpr int kUpdateOutputElems = kCudaBlockElems;

// Compatibility aliases for older headers under ProcessUnit. The active design
// uses the CUDA-style block constants above.
constexpr int kRowDataElems = kCudaBlockElems;
constexpr int kRowPadElems = 0;
constexpr int kRowPhysElems = kRowDataElems + kRowPadElems;

constexpr int kStagePrepare = 0;
constexpr int kStageReduce = 1;
constexpr int kStageSrad = 2;
constexpr int kStageSrad2 = 3;

constexpr int kCoeffPlaneJc = 0 * kCudaBlockElems;
constexpr int kCoeffPlaneJn = 1 * kCudaBlockElems;
constexpr int kCoeffPlaneJs = 2 * kCudaBlockElems;
constexpr int kCoeffPlaneJw = 3 * kCudaBlockElems;
constexpr int kCoeffPlaneJe = 4 * kCudaBlockElems;

constexpr int kUpdatePlaneI = 0 * kCudaBlockElems;
constexpr int kUpdatePlaneDn = 1 * kCudaBlockElems;
constexpr int kUpdatePlaneDs = 2 * kCudaBlockElems;
constexpr int kUpdatePlaneDw = 3 * kCudaBlockElems;
constexpr int kUpdatePlaneDe = 4 * kCudaBlockElems;
constexpr int kUpdatePlaneCn = 5 * kCudaBlockElems;
constexpr int kUpdatePlaneCs = 6 * kCudaBlockElems;
constexpr int kUpdatePlaneCe = 7 * kCudaBlockElems;

constexpr int kInputObjectFifoDepth = 2;
constexpr int kOutputObjectFifoDepth = 2;

constexpr int kAieTileCol = 1;
constexpr int kPrepareTileRow = 0;
constexpr int kReduceTileRow = 1;
constexpr int kCoeffTileRow = 2;
constexpr int kUpdateTileRow = 3;

static_assert((kCudaBlockElems % kLanes) == 0,
              "CUDA block size must be divisible by vector lanes");
static_assert((kBoardPixels % kCudaBlockElems) == 0,
              "4000x4000 image should map to full 512-element blocks");
static_assert((kReduceOutputElems * kScalarBytes) == 8,
              "reduce output is one 64-bit PLIO beat: sum and sum2");
static_assert((kMetaPacketElems % 2) == 0,
              "metadata packets must align to 64-bit PLIO words");
static_assert(kRowPhysElems == kCudaBlockElems,
              "legacy row aliases must remain block-sized");
static_assert(kCoeffInputElems == (5 * kCudaBlockElems),
              "coeff graph consumes jc/jn/js/jw/je planes");
static_assert(kUpdateInputElems == (8 * kCudaBlockElems),
              "update graph consumes image, dN/dS/dW/dE, cN/cS/cE planes");

}  // namespace srad_cfg
