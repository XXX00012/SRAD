#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

// Spill PL shim:
//   DDR image -> PL q0sqr -> q0-in-padding J tile packets for AIE ->
//   PLIO result -> DDR.

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kInputRowElems % 2 == 0,
              "64-bit PLIO packing requires an even input row");
static_assert(srad_cfg::kOutputSampleElems % 2 == 0,
              "64-bit PLIO packing requires an even output tile");

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

int tile_start(int tile_idx, int stride) {
    return tile_idx * stride;
}

void write_j_tile_row(const float* image,
                      hls::stream<plio_word_t>& out_j,
                      int tile_row_start,
                      int tile_col_start,
                      int local_row,
                      float q0sqr) {
    const int image_row =
        tile_row_start + local_row - srad_cfg::kHaloTopRows;

    for (int c = 0; c < srad_cfg::kInputRowElems; c += 2) {
#pragma HLS PIPELINE II=1
        float v0 = 0.0f;
        float v1 = 0.0f;

        if (image_row >= 0 && image_row < srad_cfg::kRows) {
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
                      int tile_col_start) {
    for (int r = 0; r < srad_cfg::kOutputTileRows; ++r) {
        for (int c = 0; c < srad_cfg::kOutputTileCols; c += 2) {
#pragma HLS PIPELINE II=1
            const plio_word_t word = in_j_next.read();
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

void write_all_tiles(const float* image,
                     hls::stream<plio_word_t>& out_j,
                     float q0sqr) {
#pragma HLS INLINE off
    for (int tile_r = 0; tile_r < srad_cfg::kTileRowCount; ++tile_r) {
        const int tile_row_start =
            tile_start(tile_r, srad_cfg::kTileStrideRows);

        for (int tile_c = 0; tile_c < srad_cfg::kTileColCount; ++tile_c) {
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
                                 q0sqr);
            }
        }
    }
}

void read_all_tiles(float* output,
                    hls::stream<plio_word_t>& in_j_next) {
#pragma HLS INLINE off
    for (int tile_r = 0; tile_r < srad_cfg::kTileRowCount; ++tile_r) {
        const int tile_row_start =
            tile_start(tile_r, srad_cfg::kTileStrideRows);

        for (int tile_c = 0; tile_c < srad_cfg::kTileColCount; ++tile_c) {
            const int tile_col_start =
                tile_start(tile_c, srad_cfg::kTileStrideCols);

            read_j_next_tile(output,
                             in_j_next,
                             tile_row_start,
                             tile_col_start);
        }
    }
}

} // namespace

extern "C" {

void TopPL(const float* image,
           float* output,
           hls::stream<plio_word_t>& out_j,
           hls::stream<plio_word_t>& in_j_next) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float sum = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = image[i];
        sum += v;
        sum2 += v * v;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    const float q0sqr = (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;

#pragma HLS DATAFLOW
    write_all_tiles(image, out_j, q0sqr);
    read_all_tiles(output, in_j_next);
}

}
