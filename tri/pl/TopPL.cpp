#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kTopPlWorkers == 1,
              "ours_1lane board path expects a single TopPL CU");
static_assert(srad_cfg::kParallelLanes == 1,
              "ours_1lane board path expects one AIE row-stream lane");
static_assert((srad_cfg::kRowPhysElems % 2) == 0,
              "64-bit PLIO packing requires an even physical row");
static_assert(srad_cfg::kQ0PadIndex == srad_cfg::kRowDataElems,
              "q0sqr is expected in the first row padding element");
static_assert(srad_cfg::kStatSumPadIndex == srad_cfg::kRowDataElems,
              "K3 writes row sum in the first output padding element");
static_assert(srad_cfg::kStatSum2PadIndex == srad_cfg::kRowDataElems + 1,
              "K3 writes row sum2 in the second output padding element");
static_assert(srad_cfg::kBoardCols ==
                  srad_cfg::kBoardStrips * srad_cfg::kRowDataElems,
              "board strips must exactly cover the image width");

constexpr int kWordsPerRow = srad_cfg::kRowPhysElems / 2;
constexpr int kWordsPerBoardStrip =
    srad_cfg::kBoardRowsPerStrip * kWordsPerRow;

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

int active_iterations(int iter_cnt) {
#pragma HLS INLINE
    int active_iters = iter_cnt;
    if (active_iters < 1) {
        active_iters = 1;
    }
    if (active_iters > srad_cfg::kBoardIterations) {
        active_iters = srad_cfg::kBoardIterations;
    }
    return active_iters;
}

float compute_q0sqr_from_sums(float sum, float sum2) {
#pragma HLS INLINE
    const float pixels = static_cast<float>(srad_cfg::kBoardPixels);
    const float mean = sum / pixels;
    const float variance = (sum2 / pixels) - (mean * mean);
    return (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;
}

void compute_initial_stats(const float* image, float& sum, float& sum2) {
#pragma HLS INLINE off
    sum = 0.0f;
    sum2 = 0.0f;

    for (int r = 0; r < srad_cfg::kBoardRows; ++r) {
        for (int c = 0; c < srad_cfg::kBoardCols; ++c) {
#pragma HLS PIPELINE II=1
            const float value = image[r * srad_cfg::kBoardCols + c];
            sum += value;
            sum2 += value * value;
        }
    }
}

float read_strip_input(const float* image,
                       int strip,
                       int stream_row,
                       int physical_col,
                       float q0sqr) {
#pragma HLS INLINE
    if (stream_row >= srad_cfg::kBoardRows) {
        return 0.0f;
    }
    if (physical_col == srad_cfg::kQ0PadIndex) {
        return q0sqr;
    }
    if (physical_col >= srad_cfg::kRowDataElems) {
        return 0.0f;
    }

    const int board_col = strip * srad_cfg::kRowDataElems + physical_col;
    return image[stream_row * srad_cfg::kBoardCols + board_col];
}

plio_word_t make_input_word(const float* image,
                            int strip,
                            int stream_row,
                            int physical_col,
                            float q0sqr) {
#pragma HLS INLINE
    const float v0 =
        read_strip_input(image, strip, stream_row, physical_col, q0sqr);
    const float v1 =
        read_strip_input(image, strip, stream_row, physical_col + 1, q0sqr);
    return pack_two_floats(v0, v1);
}

void prepare_input_rows(const float* current,
                        int strip,
                        float q0sqr,
                        hls::stream<plio_word_t>& to_aie_words) {
#pragma HLS INLINE off
    for (int stream_row = 0;
         stream_row < srad_cfg::kBoardRowsPerStrip;
         ++stream_row) {
        for (int physical_col = 0;
             physical_col < srad_cfg::kRowPhysElems;
             physical_col += 2) {
#pragma HLS PIPELINE II=1
            to_aie_words.write(make_input_word(current,
                                               strip,
                                               stream_row,
                                               physical_col,
                                               q0sqr));
        }
    }
}

void forward_input_words(hls::stream<plio_word_t>& to_aie_words,
                         hls::stream<plio_word_t>& out_j) {
#pragma HLS INLINE off
    for (int i = 0; i < kWordsPerBoardStrip; ++i) {
#pragma HLS PIPELINE II=1
        out_j.write(to_aie_words.read());
    }
}

void capture_output_words(hls::stream<plio_word_t>& in_j_next,
                          hls::stream<plio_word_t>& from_aie_words) {
#pragma HLS INLINE off
    for (int i = 0; i < kWordsPerBoardStrip; ++i) {
#pragma HLS PIPELINE II=1
        from_aie_words.write(in_j_next.read());
    }
}

float word_value_at(plio_word_t word, int physical_col, int target_col) {
#pragma HLS INLINE
    if (physical_col == target_col) {
        return unpack_lane0(word);
    }
    if ((physical_col + 1) == target_col) {
        return unpack_lane1(word);
    }
    return 0.0f;
}

void store_output_rows(float* next,
                       int strip,
                       float& next_sum,
                       float& next_sum2,
                       hls::stream<plio_word_t>& from_aie_words) {
#pragma HLS INLINE off
    const int strip_col_base = strip * srad_cfg::kRowDataElems;

    for (int stream_row = 0;
         stream_row < srad_cfg::kBoardRowsPerStrip;
         ++stream_row) {
        const int board_row = stream_row - srad_cfg::kCenterRowLag;
        const bool valid_output =
            (stream_row >= srad_cfg::kCenterRowLag) &&
            (board_row < srad_cfg::kBoardRows);
        float row_sum = 0.0f;
        float row_sum2 = 0.0f;

        for (int physical_col = 0;
             physical_col < srad_cfg::kRowPhysElems;
             physical_col += 2) {
#pragma HLS PIPELINE II=1
            const plio_word_t word = from_aie_words.read();
            const float v0 = unpack_lane0(word);
            const float v1 = unpack_lane1(word);

            if (valid_output) {
                if (physical_col < srad_cfg::kRowDataElems) {
                    next[board_row * srad_cfg::kBoardCols +
                         strip_col_base + physical_col] = v0;
                }
                if ((physical_col + 1) < srad_cfg::kRowDataElems) {
                    next[board_row * srad_cfg::kBoardCols +
                         strip_col_base + physical_col + 1] = v1;
                }

                row_sum += word_value_at(word,
                                         physical_col,
                                         srad_cfg::kStatSumPadIndex);
                row_sum2 += word_value_at(word,
                                          physical_col,
                                          srad_cfg::kStatSum2PadIndex);
            }
        }

        if (valid_output) {
            next_sum += row_sum;
            next_sum2 += row_sum2;
        }
    }
}

void run_one_strip(const float* current,
                   float* next,
                   hls::stream<plio_word_t>& out_j,
                   hls::stream<plio_word_t>& in_j_next,
                   int strip,
                   float q0sqr,
                   float& next_sum,
                   float& next_sum2) {
#pragma HLS INLINE off
    hls::stream<plio_word_t> to_aie_words;
    hls::stream<plio_word_t> from_aie_words;
#pragma HLS STREAM variable=to_aie_words depth=128
#pragma HLS STREAM variable=from_aie_words depth=128

#pragma HLS DATAFLOW
    prepare_input_rows(current, strip, q0sqr, to_aie_words);
    forward_input_words(to_aie_words, out_j);
    capture_output_words(in_j_next, from_aie_words);
    store_output_rows(next, strip, next_sum, next_sum2, from_aie_words);
}

void run_all_strips(const float* current,
                    float* next,
                    hls::stream<plio_word_t>& out_j,
                    hls::stream<plio_word_t>& in_j_next,
                    float q0sqr,
                    float& next_sum,
                    float& next_sum2) {
#pragma HLS INLINE off
    next_sum = 0.0f;
    next_sum2 = 0.0f;

    for (int strip = 0; strip < srad_cfg::kBoardStrips; ++strip) {
        run_one_strip(current,
                      next,
                      out_j,
                      in_j_next,
                      strip,
                      q0sqr,
                      next_sum,
                      next_sum2);
    }
}

void copy_final_image_to_output(const float* src, float* output) {
#pragma HLS INLINE off
    for (int i = 0; i < srad_cfg::kBoardPixels; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = src[i];
    }
}

} // namespace

extern "C" {

void TopPL(float* image,
           float* output,
           int iter_cnt,
           hls::stream<plio_word_t>& out_j,
           hls::stream<plio_word_t>& in_j_next) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int active_iters = active_iterations(iter_cnt);

    float sum = 0.0f;
    float sum2 = 0.0f;
    compute_initial_stats(image, sum, sum2);

    for (int iter = 0; iter < srad_cfg::kBoardIterations; ++iter) {
        if (iter < active_iters) {
            const float q0sqr = compute_q0sqr_from_sums(sum, sum2);
            float next_sum = 0.0f;
            float next_sum2 = 0.0f;

            if ((iter & 1) == 0) {
                run_all_strips(image,
                               output,
                               out_j,
                               in_j_next,
                               q0sqr,
                               next_sum,
                               next_sum2);
            } else {
                run_all_strips(output,
                               image,
                               out_j,
                               in_j_next,
                               q0sqr,
                               next_sum,
                               next_sum2);
            }
            sum = next_sum;
            sum2 = next_sum2;
        }
    }

    if ((active_iters & 1) == 0) {
        copy_final_image_to_output(image, output);
    }
}

}
