#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kWorkerLanes == 1,
              "ours TopPL validation build expects one lane per worker");
static_assert(srad_cfg::kInputRowElems % 2 == 0,
              "64-bit PLIO packing requires an even input row");
static_assert(srad_cfg::kOutputSampleElems % 2 == 0,
              "64-bit PLIO packing requires an even output tile");
static_assert(srad_cfg::kOutputTileDataElems == srad_cfg::kOutputTilePixels,
              "Output tile data must stay as the leading 16x16 payload");
static_assert(srad_cfg::kOutputElems == srad_cfg::kPixels,
              "Ping-pong iterations require compact output to cover the full image");

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

int clamp_worker_id(int worker_id) {
#pragma HLS INLINE
    if (worker_id < 0) {
        return 0;
    }
    if (worker_id >= srad_cfg::kTopPlWorkers) {
        return srad_cfg::kTopPlWorkers - 1;
    }
    return worker_id;
}

int tile_start(int tile_idx, int stride) {
#pragma HLS INLINE
    return tile_idx * stride;
}

int tile_linear_from_slot(int graph_iter, int worker_id) {
#pragma HLS INLINE
    return graph_iter * srad_cfg::kParallelLanes + worker_id;
}

bool is_valid_tile_linear(int tile_linear) {
#pragma HLS INLINE
    return tile_linear < srad_cfg::kTotalTileCount;
}

int tile_row_from_linear(int tile_linear) {
#pragma HLS INLINE
    return tile_linear / srad_cfg::kTileColCount;
}

int tile_col_from_linear(int tile_linear) {
#pragma HLS INLINE
    return tile_linear % srad_cfg::kTileColCount;
}

void tile_origin(int tile_linear,
                 bool& valid_tile,
                 int& tile_row_start,
                 int& tile_col_start) {
#pragma HLS INLINE
    valid_tile = is_valid_tile_linear(tile_linear);
    const int tile_r = valid_tile ? tile_row_from_linear(tile_linear) : 0;
    const int tile_c = valid_tile ? tile_col_from_linear(tile_linear) : 0;
    tile_row_start = tile_start(tile_r, srad_cfg::kTileStrideRows);
    tile_col_start = tile_start(tile_c, srad_cfg::kTileStrideCols);
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

void write_one_tile(const float* image,
                    hls::stream<plio_word_t>& out_j,
                    int tile_linear,
                    float q0sqr) {
#pragma HLS INLINE off
    bool valid_tile = false;
    int tile_row_start = 0;
    int tile_col_start = 0;
    tile_origin(tile_linear, valid_tile, tile_row_start, tile_col_start);

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
                const float v0 = unpack_lane0(word);
                const float v1 = unpack_lane1(word);
                maybe_store_valid(output,
                                  tile_row_start,
                                  tile_col_start,
                                  r,
                                  c,
                                  v0);
                maybe_store_valid(output,
                                  tile_row_start,
                                  tile_col_start,
                                  r,
                                  c + 1,
                                  v1);
            }
        }
    }

    const plio_word_t stat_word = in_j_next.read();
    if (valid_tile) {
        tile_sum = unpack_lane0(stat_word);
        tile_sum2 = unpack_lane1(stat_word);
    }
}

void read_one_tile(float* output,
                   hls::stream<plio_word_t>& in_j_next,
                   int tile_linear,
                   float& tile_sum,
                   float& tile_sum2) {
#pragma HLS INLINE off
    bool valid_tile = false;
    int tile_row_start = 0;
    int tile_col_start = 0;
    tile_origin(tile_linear, valid_tile, tile_row_start, tile_col_start);

    read_j_next_tile(output,
                     in_j_next,
                     tile_row_start,
                     tile_col_start,
                     valid_tile,
                     tile_sum,
                     tile_sum2);
}

void write_worker_tiles(const float* image,
                        hls::stream<plio_word_t>& out_j,
                        int worker_id,
                        float q0sqr) {
#pragma HLS INLINE off
    for (int graph_iter = 0;
         graph_iter < srad_cfg::kGraphRunIterations;
         ++graph_iter) {
        const int tile_linear = tile_linear_from_slot(graph_iter, worker_id);
        write_one_tile(image, out_j, tile_linear, q0sqr);
    }
}

void read_worker_tiles(float* output,
                       hls::stream<plio_word_t>& in_j_next,
                       int worker_id,
                       float& sum,
                       float& sum2) {
#pragma HLS INLINE off
    sum = 0.0f;
    sum2 = 0.0f;

    for (int graph_iter = 0;
         graph_iter < srad_cfg::kGraphRunIterations;
         ++graph_iter) {
        const int tile_linear = tile_linear_from_slot(graph_iter, worker_id);
        float tile_sum = 0.0f;
        float tile_sum2 = 0.0f;
        read_one_tile(output,
                      in_j_next,
                      tile_linear,
                      tile_sum,
                      tile_sum2);
        sum += tile_sum;
        sum2 += tile_sum2;
    }
}

void run_one_iteration(const float* image,
                       float* output,
                       hls::stream<plio_word_t>& out_j,
                       hls::stream<plio_word_t>& in_j_next,
                       int worker_id,
                       float q0sqr,
                       float& next_sum,
                       float& next_sum2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    write_worker_tiles(image, out_j, worker_id, q0sqr);
    read_worker_tiles(output, in_j_next, worker_id, next_sum, next_sum2);
}

void accumulate_image_tile(const float* image,
                           int tile_linear,
                           float& sum,
                           float& sum2) {
#pragma HLS INLINE off
    bool valid_tile = false;
    int tile_row_start = 0;
    int tile_col_start = 0;
    tile_origin(tile_linear, valid_tile, tile_row_start, tile_col_start);
    if (!valid_tile) {
        return;
    }

    for (int r = 0; r < srad_cfg::kOutputTileRows; ++r) {
        for (int c = 0; c < srad_cfg::kOutputTileCols; ++c) {
#pragma HLS PIPELINE II=1
            const int global_row = tile_row_start + r;
            const int global_col = tile_col_start + c;
            if (global_row >= 0 && global_row < srad_cfg::kRows &&
                global_col >= 0 && global_col < srad_cfg::kCols) {
                const float v =
                    image[global_row * srad_cfg::kCols + global_col];
                sum += v;
                sum2 += v * v;
            }
        }
    }
}

void compute_initial_worker_stats(const float* image,
                                  int worker_id,
                                  float& sum,
                                  float& sum2) {
#pragma HLS INLINE off
    sum = 0.0f;
    sum2 = 0.0f;

    for (int graph_iter = 0;
         graph_iter < srad_cfg::kGraphRunIterations;
         ++graph_iter) {
        const int tile_linear = tile_linear_from_slot(graph_iter, worker_id);
        accumulate_image_tile(image, tile_linear, sum, sum2);
    }
}

void copy_one_tile(const float* src, float* dst, int tile_linear) {
#pragma HLS INLINE off
    bool valid_tile = false;
    int tile_row_start = 0;
    int tile_col_start = 0;
    tile_origin(tile_linear, valid_tile, tile_row_start, tile_col_start);
    if (!valid_tile) {
        return;
    }

    for (int r = 0; r < srad_cfg::kOutputTileRows; ++r) {
        for (int c = 0; c < srad_cfg::kOutputTileCols; ++c) {
#pragma HLS PIPELINE II=1
            const int global_row = tile_row_start + r;
            const int global_col = tile_col_start + c;
            if (global_row >= 0 && global_row < srad_cfg::kRows &&
                global_col >= 0 && global_col < srad_cfg::kCols) {
                const int idx = global_row * srad_cfg::kCols + global_col;
                dst[idx] = src[idx];
            }
        }
    }
}

void copy_worker_tiles(const float* src, float* dst, int worker_id) {
#pragma HLS INLINE off
    for (int graph_iter = 0;
         graph_iter < srad_cfg::kGraphRunIterations;
         ++graph_iter) {
        const int tile_linear = tile_linear_from_slot(graph_iter, worker_id);
        copy_one_tile(src, dst, tile_linear);
    }
}

int active_iterations(int iter_cnt) {
#pragma HLS INLINE
    int active_iters = iter_cnt;
    if (active_iters < 1) {
        active_iters = 1;
    }
    if (active_iters > srad_cfg::kSradIterations) {
        active_iters = srad_cfg::kSradIterations;
    }
    return active_iters;
}

int debug_status_index(int worker_id) {
#pragma HLS INLINE
    return srad_cfg::kOutputElems + worker_id;
}

void write_debug_status(float* output, int worker_id, float status) {
#pragma HLS INLINE
    output[debug_status_index(worker_id)] = status;
}

} // namespace

extern "C" {

void TopPL(float* image,
           float* output,
           int iter_cnt,
           int worker_id_arg,
           hls::stream<plio_word_t>& out_j,
           hls::stream<plio_word_t>& in_j_next,
           hls::stream<plio_word_t>& stat_to_q0,
           hls::stream<plio_word_t>& q0_from_ctrl) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE axis port=stat_to_q0
#pragma HLS INTERFACE axis port=q0_from_ctrl
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE s_axilite port=worker_id_arg bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int worker_id = clamp_worker_id(worker_id_arg);
    const int active_iters = active_iterations(iter_cnt);
    write_debug_status(output, worker_id, 10.0f);

    float sum = 0.0f;
    float sum2 = 0.0f;
    compute_initial_worker_stats(image, worker_id, sum, sum2);
    write_debug_status(output, worker_id, 20.0f);
    stat_to_q0.write(pack_two_floats(sum, sum2));
    write_debug_status(output, worker_id, 30.0f);

    for (int iter = 0; iter < srad_cfg::kSradIterations; ++iter) {
        if (iter < active_iters) {
            const float q0sqr = unpack_lane0(q0_from_ctrl.read());
            write_debug_status(output,
                               worker_id,
                               40.0f + static_cast<float>(iter));
            float next_sum = 0.0f;
            float next_sum2 = 0.0f;

            write_debug_status(output,
                               worker_id,
                               50.0f + static_cast<float>(iter));
            if ((iter & 1) == 0) {
                run_one_iteration(image,
                                  output,
                                  out_j,
                                  in_j_next,
                                  worker_id,
                                  q0sqr,
                                  next_sum,
                                  next_sum2);
            } else {
                run_one_iteration(output,
                                  image,
                                  out_j,
                                  in_j_next,
                                  worker_id,
                                  q0sqr,
                                  next_sum,
                                  next_sum2);
            }
            write_debug_status(output,
                               worker_id,
                               60.0f + static_cast<float>(iter));

            stat_to_q0.write(pack_two_floats(next_sum, next_sum2));
            write_debug_status(output,
                               worker_id,
                               70.0f + static_cast<float>(iter));
        }
    }

    if ((active_iters & 1) == 0) {
        write_debug_status(output, worker_id, 80.0f);
        copy_worker_tiles(image, output, worker_id);
    }
    write_debug_status(output, worker_id, 90.0f);
}

}
