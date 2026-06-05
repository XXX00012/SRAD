#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

// Ours_400 PL shim:
//   DDR image -> PL q0sqr -> 200 q0-in-padding J tile streams for AIE ->
//   100 pktmerge<2> result streams + tile stats -> DDR. Multi-iteration runs
//   ping-pong between image/output buffers; q0sqr after the first iteration
//   comes from AIE tile stats.

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kInputRowElems % 2 == 0,
              "64-bit PLIO packing requires an even input row");
static_assert(srad_cfg::kOutputSampleElems % 2 == 0,
              "64-bit PLIO packing requires an even output tile");
static_assert(srad_cfg::kOutputTileDataElems == srad_cfg::kOutputTilePixels,
              "Output tile data must stay as the leading 16x16 payload");
static_assert(srad_cfg::kOutputElems == srad_cfg::kPixels,
              "Ping-pong iterations require compact output to cover the full image");
static_assert(srad_cfg::kRows % srad_cfg::kTileStrideRows == 0,
              "Tile stats q0 path expects no partial output tile rows");
static_assert(srad_cfg::kCols % srad_cfg::kTileStrideCols == 0,
              "Tile stats q0 path expects no partial output tile cols");
static_assert(srad_cfg::kOutputPlioGroups * srad_cfg::kOutputLanesPerPlio ==
                  srad_cfg::kParallelLanes,
              "Grouped output streams must cover every parallel lane");

ap_uint<32> float_to_bits(float value) {
    union {
        float f;
        uint32_t u;
    } conv;

    conv.f = value;
    return ap_uint<32>(conv.u);
}

float bits_to_float(ap_uint<32> bits) {
    union {
        float f;
        uint32_t u;
    } conv;

    conv.u = static_cast<uint32_t>(bits);
    return conv.f;
}

plio_word_t pack_two_floats(float lane0, float lane1) {
    plio_word_t word = 0;
    word.range(31, 0) = float_to_bits(lane0);
    word.range(63, 32) = float_to_bits(lane1);
    return word;
}

float unpack_lane0(plio_word_t word) {
    return bits_to_float(word.range(31, 0));
}

float unpack_lane1(plio_word_t word) {
    return bits_to_float(word.range(63, 32));
}

float compute_q0sqr_from_sums(float sum, float sum2) {
#pragma HLS INLINE
    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    return (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;
}

int tile_start(int tile_idx, int stride) {
    return tile_idx * stride;
}

int tile_linear_from_slot(int iteration, int lane) {
    return iteration * srad_cfg::kParallelLanes + lane;
}

bool is_valid_tile_linear(int tile_linear) {
    return tile_linear < srad_cfg::kTotalTileCount;
}

int tile_row_from_linear(int tile_linear) {
    return tile_linear / srad_cfg::kTileColCount;
}

int tile_col_from_linear(int tile_linear) {
    return tile_linear % srad_cfg::kTileColCount;
}

void write_j_tile_row(const float* image,
                      hls::stream<plio_word_t>& out_j,
                      int tile_row_start,
                      int tile_col_start,
                      int local_row,
                      float q0sqr,
                      bool valid_tile) {
    const int image_row =
        tile_row_start + local_row - srad_cfg::kHaloTopRows;

    for (int c = 0; c < srad_cfg::kInputRowElems; c += 2) {
#pragma HLS PIPELINE II=1
        float v0 = 0.0f;
        float v1 = 0.0f;

        if (valid_tile &&
            image_row >= 0 && image_row < srad_cfg::kRows) {
            if (c < srad_cfg::kInputLogicalCols) {
                const int image_col0 =
                    tile_col_start + c - srad_cfg::kHaloLeftCols;
                if (image_col0 >= 0 && image_col0 < srad_cfg::kCols) {
                    v0 = image[image_row * srad_cfg::kCols + image_col0];
                }
            }

            if ((c + 1) < srad_cfg::kInputLogicalCols) {
                const int image_col1 =
                    tile_col_start + c + 1 - srad_cfg::kHaloLeftCols;
                if (image_col1 >= 0 && image_col1 < srad_cfg::kCols) {
                    v1 = image[image_row * srad_cfg::kCols + image_col1];
                }
            }
        }

        if (local_row == srad_cfg::kQ0SqrTileRow &&
            c == srad_cfg::kQ0SqrTileCol) {
            v0 = q0sqr;
        }
        if (local_row == srad_cfg::kQ0SqrTileRow &&
            (c + 1) == srad_cfg::kQ0SqrTileCol) {
            v1 = q0sqr;
        }

        out_j.write(pack_two_floats(v0, v1));
    }
}

void maybe_store_valid(float* output,
                       int tile_row_start,
                       int tile_col_start,
                       int local_row,
                       int local_col,
                       float value) {
#pragma HLS INLINE
    const int global_row = tile_row_start + local_row;
    const int global_col = tile_col_start + local_col;
    if (global_row < srad_cfg::kOutputFirstRow ||
        global_row > srad_cfg::kOutputLastRow ||
        global_col < srad_cfg::kOutputFirstCol ||
        global_col > srad_cfg::kOutputLastCol) {
        return;
    }

    const int output_row = global_row - srad_cfg::kOutputFirstRow;
    const int output_col = global_col - srad_cfg::kOutputFirstCol;
    output[output_row * srad_cfg::kOutputCols + output_col] = value;
}

void read_j_next_tile(float* output,
                      hls::stream<plio_word_t>& in_j_next,
                      int tile_row_start,
                      int tile_col_start,
                      bool valid_tile,
                      float& tile_sum,
                      float& tile_sum2) {
    tile_sum = 0.0f;
    tile_sum2 = 0.0f;
    for (int r = 0; r < srad_cfg::kOutputTileRows; ++r) {
        for (int c = 0; c < srad_cfg::kOutputTileCols; c += 2) {
#pragma HLS PIPELINE II=1
            const plio_word_t word = in_j_next.read();
            if (valid_tile) {
                maybe_store_valid(output,
                                  tile_row_start,
                                  tile_col_start,
                                  r,
                                  c,
                                  unpack_lane0(word));
                maybe_store_valid(output,
                                  tile_row_start,
                                  tile_col_start,
                                  r,
                                  c + 1,
                                  unpack_lane1(word));
            }
        }
    }

    const plio_word_t stat_word = in_j_next.read();
    tile_sum = unpack_lane0(stat_word);
    tile_sum2 = unpack_lane1(stat_word);
}

void write_one_tile(const float* image,
                    hls::stream<plio_word_t>& out_j,
                    int tile_linear,
                    float q0sqr) {
#pragma HLS INLINE off
    const bool valid_tile = is_valid_tile_linear(tile_linear);
    const int tile_r = valid_tile ? tile_row_from_linear(tile_linear) : 0;
    const int tile_c = valid_tile ? tile_col_from_linear(tile_linear) : 0;
    const int tile_row_start =
        tile_start(tile_r, srad_cfg::kTileStrideRows);
    const int tile_col_start =
        tile_start(tile_c, srad_cfg::kTileStrideCols);

    for (int local_row = 0;
         local_row < srad_cfg::kInputLogicalRows;
         ++local_row) {
        write_j_tile_row(image,
                         out_j,
                         tile_row_start,
                         tile_col_start,
                         local_row,
                         q0sqr,
                         valid_tile);
    }
}

void write_all_lane_tiles(
    const float* image,
    hls::stream<plio_word_t> out_j[srad_cfg::kParallelLanes],
    float q0sqr) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
    for (int it = 0; it < srad_cfg::kGraphRunIterations; ++it) {
        for (int lane = 0; lane < srad_cfg::kParallelLanes; ++lane) {
            const int tile_linear = tile_linear_from_slot(it, lane);
            write_one_tile(image, out_j[lane], tile_linear, q0sqr);
        }
    }
}

void read_one_tile(float* output,
                   hls::stream<plio_word_t>& in_j_next,
                   int tile_linear,
                   float& tile_sum,
                   float& tile_sum2,
                   bool& valid_tile) {
#pragma HLS INLINE off
    valid_tile = is_valid_tile_linear(tile_linear);
    const int tile_r = valid_tile ? tile_row_from_linear(tile_linear) : 0;
    const int tile_c = valid_tile ? tile_col_from_linear(tile_linear) : 0;
    const int tile_row_start =
        tile_start(tile_r, srad_cfg::kTileStrideRows);
    const int tile_col_start =
        tile_start(tile_c, srad_cfg::kTileStrideCols);

    read_j_next_tile(output,
                     in_j_next,
                     tile_row_start,
                     tile_col_start,
                     valid_tile,
                     tile_sum,
                     tile_sum2);
}

void read_all_group_tiles(
    float* output,
    hls::stream<plio_word_t> in_j_next[srad_cfg::kOutputPlioGroups],
    float& sum,
    float& sum2) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
    sum = 0.0f;
    sum2 = 0.0f;

    for (int it = 0; it < srad_cfg::kGraphRunIterations; ++it) {
        for (int group = 0; group < srad_cfg::kOutputPlioGroups; ++group) {
            for (int local = 0;
                 local < srad_cfg::kOutputLanesPerPlio;
                 ++local) {
                const int lane =
                    group * srad_cfg::kOutputLanesPerPlio + local;
                const int tile_linear = tile_linear_from_slot(it, lane);
                float tile_sum = 0.0f;
                float tile_sum2 = 0.0f;
                bool valid_tile = false;

                read_one_tile(output,
                              in_j_next[group],
                              tile_linear,
                              tile_sum,
                              tile_sum2,
                              valid_tile);
                if (valid_tile) {
                    sum += tile_sum;
                    sum2 += tile_sum2;
                }
            }
        }
    }
}

float compute_initial_q0sqr(const float* image) {
#pragma HLS INLINE off
    float sum = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = image[i];
        sum += v;
        sum2 += v * v;
    }

    return compute_q0sqr_from_sums(sum, sum2);
}

void run_one_iteration(
    const float* image,
    float* output,
    hls::stream<plio_word_t> out_j[srad_cfg::kParallelLanes],
    hls::stream<plio_word_t> in_j_next[srad_cfg::kOutputPlioGroups],
    float q0sqr,
    float& next_sum,
    float& next_sum2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    write_all_lane_tiles(image, out_j, q0sqr);
    read_all_group_tiles(output, in_j_next, next_sum, next_sum2);
}

void copy_image(const float* src, float* dst) {
#pragma HLS INLINE off
    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

} // namespace

extern "C" {

void TopPL(float* image,
           float* output,
           int iter_cnt,
           hls::stream<plio_word_t> out_j[srad_cfg::kParallelLanes],
           hls::stream<plio_word_t> in_j_next[srad_cfg::kOutputPlioGroups]) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int active_iters = iter_cnt;
    if (active_iters < 1) {
        active_iters = 1;
    }
    if (active_iters > srad_cfg::kSradIterations) {
        active_iters = srad_cfg::kSradIterations;
    }

    float q0sqr = compute_initial_q0sqr(image);

    for (int iter = 0; iter < srad_cfg::kSradIterations; ++iter) {
        if (iter < active_iters) {
            float next_sum = 0.0f;
            float next_sum2 = 0.0f;

            if ((iter & 1) == 0) {
                run_one_iteration(image,
                                  output,
                                  out_j,
                                  in_j_next,
                                  q0sqr,
                                  next_sum,
                                  next_sum2);
            } else {
                run_one_iteration(output,
                                  image,
                                  out_j,
                                  in_j_next,
                                  q0sqr,
                                  next_sum,
                                  next_sum2);
            }

            q0sqr = compute_q0sqr_from_sums(next_sum, next_sum2);
        }
    }

    if ((active_iters & 1) == 0) {
        copy_image(image, output);
    }
}

}
