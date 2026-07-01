#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kTopPlWorkers == 1,
              "ours_16lane board path expects a single TopPL CU");
static_assert(srad_cfg::kParallelLanes == 16,
              "ours_16lane board path expects sixteen AIE row-stream lanes");
static_assert((srad_cfg::kRowPhysElems % 2) == 0,
              "64-bit PLIO packing requires an even physical row");
static_assert(srad_cfg::kQ0PadIndex == srad_cfg::kRowDataElems,
              "q0sqr is expected in the first row padding element");
static_assert(srad_cfg::kStatSumPadIndex == srad_cfg::kRowDataElems,
              "row sum is expected in the first row padding element");
static_assert(srad_cfg::kStatSum2PadIndex == srad_cfg::kRowDataElems + 1,
              "row sum2 is expected in the second row padding element");
static_assert((srad_cfg::kStatSumPadIndex % 2) == 1,
              "row sum is expected in PLIO lane 1");
static_assert((srad_cfg::kStatSum2PadIndex % 2) == 0,
              "row sum2 is expected in PLIO lane 0");
static_assert(srad_cfg::kBoardCols ==
                  srad_cfg::kBoardStrips * srad_cfg::kRowDataElems,
              "board strips must exactly cover the image width");

constexpr int kWordsPerRow = srad_cfg::kRowPhysElems / 2;
constexpr int kStreamRowsPerLane = srad_cfg::kBoardRowsPerLaneStream;
constexpr int kWordsPerLaneStrip = kStreamRowsPerLane * kWordsPerRow;
constexpr int kLanesPerBatch = srad_cfg::kParallelLanes;
constexpr int kStripBatches = srad_cfg::kBoardStripBatches;
constexpr int kLeadingInvalidOutputRows =
    srad_cfg::kBoardLanePreContextRows + srad_cfg::kCenterRowLag;
constexpr int kFullDataOutputWords = srad_cfg::kRowDataElems / 2;
constexpr int kLastDataPhysicalCol = kFullDataOutputWords * 2;
constexpr int kSum2PhysicalCol = kLastDataPhysicalCol + 2;
constexpr int kBatchDataElems =
    kLanesPerBatch * srad_cfg::kRowDataElems;

static_assert(kLanesPerBatch == 16,
              "TopPL column batch path expects sixteen lanes per batch");
static_assert(kStripBatches == 2,
              "4000 columns with 16 lanes of 125 columns should use two batches");
static_assert(kStreamRowsPerLane ==
                  kLeadingInvalidOutputRows + srad_cfg::kBoardRows,
              "output stream should have only leading invalid context rows");
static_assert((srad_cfg::kRowDataElems % 2) == 1,
              "output fast path expects the final data element to share a word with sum");
static_assert(kLastDataPhysicalCol == srad_cfg::kRowDataElems - 1,
              "last data word should contain the final data element");
static_assert(kLastDataPhysicalCol + 1 == srad_cfg::kStatSumPadIndex,
              "sum should share the last data word");
static_assert(kSum2PhysicalCol == srad_cfg::kStatSum2PadIndex,
              "sum2 should start the final output word");
static_assert(kWordsPerRow == kFullDataOutputWords + 2,
              "output fast path expects full data words plus sum and sum2 words");

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

void compute_initial_stats(const float* image,
                           float& sum,
                           float& sum2) {
#pragma HLS INLINE off
    sum = 0.0f;
    sum2 = 0.0f;

    for (int r = 0; r < srad_cfg::kBoardRows; ++r) {
        const int row_base = r * srad_cfg::kBoardCols;
        for (int c = 0; c < srad_cfg::kBoardCols; ++c) {
#pragma HLS PIPELINE II=1
            const float value = image[row_base + c];
            sum += value;
            sum2 += value * value;
        }
    }
}

void forward_input_words(hls::stream<plio_word_t>& to_aie_words,
                         hls::stream<plio_word_t>& out_j) {
#pragma HLS INLINE off
    for (int i = 0; i < kWordsPerLaneStrip; ++i) {
#pragma HLS PIPELINE II=1
        out_j.write(to_aie_words.read());
    }
}

void capture_output_words(hls::stream<plio_word_t>& in_j_next,
                          hls::stream<plio_word_t>& from_aie_words) {
#pragma HLS INLINE off
    for (int i = 0; i < kWordsPerLaneStrip; ++i) {
#pragma HLS PIPELINE II=1
        from_aie_words.write(in_j_next.read());
    }
}

template<int Lane>
float buffered_input_value(float row_elems[kBatchDataElems],
                           int physical_col,
                           bool valid_row,
                           float q0sqr) {
#pragma HLS INLINE
    if (!valid_row) {
        return 0.0f;
    }
    if (physical_col == srad_cfg::kQ0PadIndex) {
        return q0sqr;
    }
    if (physical_col < srad_cfg::kRowDataElems) {
        return row_elems[Lane * srad_cfg::kRowDataElems + physical_col];
    }
    return 0.0f;
}

void load_16lane_input_rows(const float* current,
                            int strip_batch,
                            float q0sqr,
                            hls::stream<plio_word_t>& to_aie_words0,
                            hls::stream<plio_word_t>& to_aie_words1,
                            hls::stream<plio_word_t>& to_aie_words2,
                            hls::stream<plio_word_t>& to_aie_words3,
                            hls::stream<plio_word_t>& to_aie_words4,
                            hls::stream<plio_word_t>& to_aie_words5,
                            hls::stream<plio_word_t>& to_aie_words6,
                            hls::stream<plio_word_t>& to_aie_words7,
                            hls::stream<plio_word_t>& to_aie_words8,
                            hls::stream<plio_word_t>& to_aie_words9,
                            hls::stream<plio_word_t>& to_aie_words10,
                            hls::stream<plio_word_t>& to_aie_words11,
                            hls::stream<plio_word_t>& to_aie_words12,
                            hls::stream<plio_word_t>& to_aie_words13,
                            hls::stream<plio_word_t>& to_aie_words14,
                            hls::stream<plio_word_t>& to_aie_words15) {
#pragma HLS INLINE off
    float row_elems[kBatchDataElems];
#pragma HLS ARRAY_PARTITION variable=row_elems cyclic factor=16 dim=1

    for (int stream_row = 0;
         stream_row < kStreamRowsPerLane;
         ++stream_row) {
        const int first_row = -srad_cfg::kBoardLanePreContextRows;
        const int board_row = first_row + stream_row;
        const bool valid_row =
            (board_row >= 0) && (board_row < srad_cfg::kBoardRows);
        const int batch_col_base =
            strip_batch * kLanesPerBatch * srad_cfg::kRowDataElems;

        if (valid_row) {
            const int row_base =
                board_row * srad_cfg::kBoardCols + batch_col_base;

            for (int elem = 0; elem < kBatchDataElems; ++elem) {
#pragma HLS PIPELINE II=1
                row_elems[elem] = current[row_base + elem];
            }
        } else {
            for (int elem = 0; elem < kBatchDataElems; ++elem) {
#pragma HLS PIPELINE II=1
                row_elems[elem] = 0.0f;
            }
        }

        for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
            const int col0 = word_col * 2;
            const int col1 = col0 + 1;
            to_aie_words0.write(pack_two_floats(
                buffered_input_value<0>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<0>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words1.write(pack_two_floats(
                buffered_input_value<1>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<1>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words2.write(pack_two_floats(
                buffered_input_value<2>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<2>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words3.write(pack_two_floats(
                buffered_input_value<3>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<3>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words4.write(pack_two_floats(
                buffered_input_value<4>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<4>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words5.write(pack_two_floats(
                buffered_input_value<5>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<5>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words6.write(pack_two_floats(
                buffered_input_value<6>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<6>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words7.write(pack_two_floats(
                buffered_input_value<7>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<7>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words8.write(pack_two_floats(
                buffered_input_value<8>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<8>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words9.write(pack_two_floats(
                buffered_input_value<9>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<9>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words10.write(pack_two_floats(
                buffered_input_value<10>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<10>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words11.write(pack_two_floats(
                buffered_input_value<11>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<11>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words12.write(pack_two_floats(
                buffered_input_value<12>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<12>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words13.write(pack_two_floats(
                buffered_input_value<13>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<13>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words14.write(pack_two_floats(
                buffered_input_value<14>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<14>(row_elems, col1, valid_row, q0sqr)));
            to_aie_words15.write(pack_two_floats(
                buffered_input_value<15>(row_elems, col0, valid_row, q0sqr),
                buffered_input_value<15>(row_elems, col1, valid_row, q0sqr)));
        }
    }
}

void run_one_lane_stream(hls::stream<plio_word_t>& to_aie_words,
                         hls::stream<plio_word_t>& out_j,
                         hls::stream<plio_word_t>& in_j_next,
                         hls::stream<plio_word_t>& from_aie_words) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    forward_input_words(to_aie_words, out_j);
    capture_output_words(in_j_next, from_aie_words);
}

template<int Lane>
void capture_lane_output_word(int word_col,
                              hls::stream<plio_word_t>& from_aie_words,
                              float row_elems[kBatchDataElems],
                              float& lane_sum,
                              float& lane_sum2) {
#pragma HLS INLINE
    const plio_word_t word = from_aie_words.read();
    const int physical_col = word_col * 2;
    const int next_physical_col = physical_col + 1;
    const int lane_base = Lane * srad_cfg::kRowDataElems;
    const float v0 = unpack_lane0(word);
    const float v1 = unpack_lane1(word);

    if (physical_col < srad_cfg::kRowDataElems) {
        row_elems[lane_base + physical_col] = v0;
    }
    if (next_physical_col < srad_cfg::kRowDataElems) {
        row_elems[lane_base + next_physical_col] = v1;
    }

    if (physical_col == srad_cfg::kStatSumPadIndex - 1) {
        lane_sum += v1;
    }
    if (physical_col == srad_cfg::kStatSum2PadIndex) {
        lane_sum2 += v0;
    }
}

void capture_16lane_output_row(hls::stream<plio_word_t>& from_aie_words0,
                               hls::stream<plio_word_t>& from_aie_words1,
                               hls::stream<plio_word_t>& from_aie_words2,
                               hls::stream<plio_word_t>& from_aie_words3,
                               hls::stream<plio_word_t>& from_aie_words4,
                               hls::stream<plio_word_t>& from_aie_words5,
                               hls::stream<plio_word_t>& from_aie_words6,
                               hls::stream<plio_word_t>& from_aie_words7,
                               hls::stream<plio_word_t>& from_aie_words8,
                               hls::stream<plio_word_t>& from_aie_words9,
                               hls::stream<plio_word_t>& from_aie_words10,
                               hls::stream<plio_word_t>& from_aie_words11,
                               hls::stream<plio_word_t>& from_aie_words12,
                               hls::stream<plio_word_t>& from_aie_words13,
                               hls::stream<plio_word_t>& from_aie_words14,
                               hls::stream<plio_word_t>& from_aie_words15,
                               float row_elems[kBatchDataElems],
                               float lane_sum[srad_cfg::kParallelLanes],
                               float lane_sum2[srad_cfg::kParallelLanes]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
        capture_lane_output_word<0>(word_col, from_aie_words0,
                                    row_elems, lane_sum[0], lane_sum2[0]);
        capture_lane_output_word<1>(word_col, from_aie_words1,
                                    row_elems, lane_sum[1], lane_sum2[1]);
        capture_lane_output_word<2>(word_col, from_aie_words2,
                                    row_elems, lane_sum[2], lane_sum2[2]);
        capture_lane_output_word<3>(word_col, from_aie_words3,
                                    row_elems, lane_sum[3], lane_sum2[3]);
        capture_lane_output_word<4>(word_col, from_aie_words4,
                                    row_elems, lane_sum[4], lane_sum2[4]);
        capture_lane_output_word<5>(word_col, from_aie_words5,
                                    row_elems, lane_sum[5], lane_sum2[5]);
        capture_lane_output_word<6>(word_col, from_aie_words6,
                                    row_elems, lane_sum[6], lane_sum2[6]);
        capture_lane_output_word<7>(word_col, from_aie_words7,
                                    row_elems, lane_sum[7], lane_sum2[7]);
        capture_lane_output_word<8>(word_col, from_aie_words8,
                                    row_elems, lane_sum[8], lane_sum2[8]);
        capture_lane_output_word<9>(word_col, from_aie_words9,
                                    row_elems, lane_sum[9], lane_sum2[9]);
        capture_lane_output_word<10>(word_col, from_aie_words10,
                                     row_elems, lane_sum[10], lane_sum2[10]);
        capture_lane_output_word<11>(word_col, from_aie_words11,
                                     row_elems, lane_sum[11], lane_sum2[11]);
        capture_lane_output_word<12>(word_col, from_aie_words12,
                                     row_elems, lane_sum[12], lane_sum2[12]);
        capture_lane_output_word<13>(word_col, from_aie_words13,
                                     row_elems, lane_sum[13], lane_sum2[13]);
        capture_lane_output_word<14>(word_col, from_aie_words14,
                                     row_elems, lane_sum[14], lane_sum2[14]);
        capture_lane_output_word<15>(word_col, from_aie_words15,
                                     row_elems, lane_sum[15], lane_sum2[15]);
    }
}

void discard_16lane_output_row(hls::stream<plio_word_t>& from_aie_words0,
                               hls::stream<plio_word_t>& from_aie_words1,
                               hls::stream<plio_word_t>& from_aie_words2,
                               hls::stream<plio_word_t>& from_aie_words3,
                               hls::stream<plio_word_t>& from_aie_words4,
                               hls::stream<plio_word_t>& from_aie_words5,
                               hls::stream<plio_word_t>& from_aie_words6,
                               hls::stream<plio_word_t>& from_aie_words7,
                               hls::stream<plio_word_t>& from_aie_words8,
                               hls::stream<plio_word_t>& from_aie_words9,
                               hls::stream<plio_word_t>& from_aie_words10,
                               hls::stream<plio_word_t>& from_aie_words11,
                               hls::stream<plio_word_t>& from_aie_words12,
                               hls::stream<plio_word_t>& from_aie_words13,
                               hls::stream<plio_word_t>& from_aie_words14,
                               hls::stream<plio_word_t>& from_aie_words15) {
#pragma HLS INLINE off
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
        (void)from_aie_words0.read();
        (void)from_aie_words1.read();
        (void)from_aie_words2.read();
        (void)from_aie_words3.read();
        (void)from_aie_words4.read();
        (void)from_aie_words5.read();
        (void)from_aie_words6.read();
        (void)from_aie_words7.read();
        (void)from_aie_words8.read();
        (void)from_aie_words9.read();
        (void)from_aie_words10.read();
        (void)from_aie_words11.read();
        (void)from_aie_words12.read();
        (void)from_aie_words13.read();
        (void)from_aie_words14.read();
        (void)from_aie_words15.read();
    }
}

void write_contiguous_output_row(float* next,
                                 int strip_batch,
                                 int board_row,
                                 float row_elems[kBatchDataElems]) {
#pragma HLS INLINE off
    const int batch_col_base =
        strip_batch * kLanesPerBatch * srad_cfg::kRowDataElems;
    const int row_base =
        board_row * srad_cfg::kBoardCols + batch_col_base;

    for (int elem = 0; elem < kBatchDataElems; ++elem) {
#pragma HLS PIPELINE II=1
        next[row_base + elem] = row_elems[elem];
    }
}

void store_16lane_output_rows(float* next,
                              int strip_batch,
                              hls::stream<plio_word_t>& from_aie_words0,
                              hls::stream<plio_word_t>& from_aie_words1,
                              hls::stream<plio_word_t>& from_aie_words2,
                              hls::stream<plio_word_t>& from_aie_words3,
                              hls::stream<plio_word_t>& from_aie_words4,
                              hls::stream<plio_word_t>& from_aie_words5,
                              hls::stream<plio_word_t>& from_aie_words6,
                              hls::stream<plio_word_t>& from_aie_words7,
                              hls::stream<plio_word_t>& from_aie_words8,
                              hls::stream<plio_word_t>& from_aie_words9,
                              hls::stream<plio_word_t>& from_aie_words10,
                              hls::stream<plio_word_t>& from_aie_words11,
                              hls::stream<plio_word_t>& from_aie_words12,
                              hls::stream<plio_word_t>& from_aie_words13,
                              hls::stream<plio_word_t>& from_aie_words14,
                              hls::stream<plio_word_t>& from_aie_words15,
                              hls::stream<plio_word_t>& lane_stat0,
                              hls::stream<plio_word_t>& lane_stat1,
                              hls::stream<plio_word_t>& lane_stat2,
                              hls::stream<plio_word_t>& lane_stat3,
                              hls::stream<plio_word_t>& lane_stat4,
                              hls::stream<plio_word_t>& lane_stat5,
                              hls::stream<plio_word_t>& lane_stat6,
                              hls::stream<plio_word_t>& lane_stat7,
                              hls::stream<plio_word_t>& lane_stat8,
                              hls::stream<plio_word_t>& lane_stat9,
                              hls::stream<plio_word_t>& lane_stat10,
                              hls::stream<plio_word_t>& lane_stat11,
                              hls::stream<plio_word_t>& lane_stat12,
                              hls::stream<plio_word_t>& lane_stat13,
                              hls::stream<plio_word_t>& lane_stat14,
                              hls::stream<plio_word_t>& lane_stat15) {
#pragma HLS INLINE off
    float row_elems[kBatchDataElems];
    float lane_sum[srad_cfg::kParallelLanes];
    float lane_sum2[srad_cfg::kParallelLanes];
#pragma HLS ARRAY_PARTITION variable=row_elems cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    for (int lane = 0; lane < srad_cfg::kParallelLanes; ++lane) {
#pragma HLS UNROLL
        lane_sum[lane] = 0.0f;
        lane_sum2[lane] = 0.0f;
    }

    for (int stream_row = 0;
         stream_row < kLeadingInvalidOutputRows;
         ++stream_row) {
        discard_16lane_output_row(from_aie_words0,
                                  from_aie_words1,
                                  from_aie_words2,
                                  from_aie_words3,
                                  from_aie_words4,
                                  from_aie_words5,
                                  from_aie_words6,
                                  from_aie_words7,
                                  from_aie_words8,
                                  from_aie_words9,
                                  from_aie_words10,
                                  from_aie_words11,
                                  from_aie_words12,
                                  from_aie_words13,
                                  from_aie_words14,
                                  from_aie_words15);
    }

    for (int board_row = 0;
         board_row < srad_cfg::kBoardRows;
         ++board_row) {
        capture_16lane_output_row(from_aie_words0,
                                  from_aie_words1,
                                  from_aie_words2,
                                  from_aie_words3,
                                  from_aie_words4,
                                  from_aie_words5,
                                  from_aie_words6,
                                  from_aie_words7,
                                  from_aie_words8,
                                  from_aie_words9,
                                  from_aie_words10,
                                  from_aie_words11,
                                  from_aie_words12,
                                  from_aie_words13,
                                  from_aie_words14,
                                  from_aie_words15,
                                  row_elems,
                                  lane_sum,
                                  lane_sum2);
        write_contiguous_output_row(next, strip_batch, board_row, row_elems);
    }

    lane_stat0.write(pack_two_floats(lane_sum[0], lane_sum2[0]));
    lane_stat1.write(pack_two_floats(lane_sum[1], lane_sum2[1]));
    lane_stat2.write(pack_two_floats(lane_sum[2], lane_sum2[2]));
    lane_stat3.write(pack_two_floats(lane_sum[3], lane_sum2[3]));
    lane_stat4.write(pack_two_floats(lane_sum[4], lane_sum2[4]));
    lane_stat5.write(pack_two_floats(lane_sum[5], lane_sum2[5]));
    lane_stat6.write(pack_two_floats(lane_sum[6], lane_sum2[6]));
    lane_stat7.write(pack_two_floats(lane_sum[7], lane_sum2[7]));
    lane_stat8.write(pack_two_floats(lane_sum[8], lane_sum2[8]));
    lane_stat9.write(pack_two_floats(lane_sum[9], lane_sum2[9]));
    lane_stat10.write(pack_two_floats(lane_sum[10], lane_sum2[10]));
    lane_stat11.write(pack_two_floats(lane_sum[11], lane_sum2[11]));
    lane_stat12.write(pack_two_floats(lane_sum[12], lane_sum2[12]));
    lane_stat13.write(pack_two_floats(lane_sum[13], lane_sum2[13]));
    lane_stat14.write(pack_two_floats(lane_sum[14], lane_sum2[14]));
    lane_stat15.write(pack_two_floats(lane_sum[15], lane_sum2[15]));
}

void collect_16lane_stats(hls::stream<plio_word_t>& lane_stat0,
                          hls::stream<plio_word_t>& lane_stat1,
                          hls::stream<plio_word_t>& lane_stat2,
                          hls::stream<plio_word_t>& lane_stat3,
                          hls::stream<plio_word_t>& lane_stat4,
                          hls::stream<plio_word_t>& lane_stat5,
                          hls::stream<plio_word_t>& lane_stat6,
                          hls::stream<plio_word_t>& lane_stat7,
                          hls::stream<plio_word_t>& lane_stat8,
                          hls::stream<plio_word_t>& lane_stat9,
                          hls::stream<plio_word_t>& lane_stat10,
                          hls::stream<plio_word_t>& lane_stat11,
                          hls::stream<plio_word_t>& lane_stat12,
                          hls::stream<plio_word_t>& lane_stat13,
                          hls::stream<plio_word_t>& lane_stat14,
                          hls::stream<plio_word_t>& lane_stat15,
                          float lane_sum[srad_cfg::kParallelLanes],
                          float lane_sum2[srad_cfg::kParallelLanes]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    const plio_word_t s0 = lane_stat0.read();
    lane_sum[0] = unpack_lane0(s0);
    lane_sum2[0] = unpack_lane1(s0);
    const plio_word_t s1 = lane_stat1.read();
    lane_sum[1] = unpack_lane0(s1);
    lane_sum2[1] = unpack_lane1(s1);
    const plio_word_t s2 = lane_stat2.read();
    lane_sum[2] = unpack_lane0(s2);
    lane_sum2[2] = unpack_lane1(s2);
    const plio_word_t s3 = lane_stat3.read();
    lane_sum[3] = unpack_lane0(s3);
    lane_sum2[3] = unpack_lane1(s3);
    const plio_word_t s4 = lane_stat4.read();
    lane_sum[4] = unpack_lane0(s4);
    lane_sum2[4] = unpack_lane1(s4);
    const plio_word_t s5 = lane_stat5.read();
    lane_sum[5] = unpack_lane0(s5);
    lane_sum2[5] = unpack_lane1(s5);
    const plio_word_t s6 = lane_stat6.read();
    lane_sum[6] = unpack_lane0(s6);
    lane_sum2[6] = unpack_lane1(s6);
    const plio_word_t s7 = lane_stat7.read();
    lane_sum[7] = unpack_lane0(s7);
    lane_sum2[7] = unpack_lane1(s7);
    const plio_word_t s8 = lane_stat8.read();
    lane_sum[8] = unpack_lane0(s8);
    lane_sum2[8] = unpack_lane1(s8);
    const plio_word_t s9 = lane_stat9.read();
    lane_sum[9] = unpack_lane0(s9);
    lane_sum2[9] = unpack_lane1(s9);
    const plio_word_t s10 = lane_stat10.read();
    lane_sum[10] = unpack_lane0(s10);
    lane_sum2[10] = unpack_lane1(s10);
    const plio_word_t s11 = lane_stat11.read();
    lane_sum[11] = unpack_lane0(s11);
    lane_sum2[11] = unpack_lane1(s11);
    const plio_word_t s12 = lane_stat12.read();
    lane_sum[12] = unpack_lane0(s12);
    lane_sum2[12] = unpack_lane1(s12);
    const plio_word_t s13 = lane_stat13.read();
    lane_sum[13] = unpack_lane0(s13);
    lane_sum2[13] = unpack_lane1(s13);
    const plio_word_t s14 = lane_stat14.read();
    lane_sum[14] = unpack_lane0(s14);
    lane_sum2[14] = unpack_lane1(s14);
    const plio_word_t s15 = lane_stat15.read();
    lane_sum[15] = unpack_lane0(s15);
    lane_sum2[15] = unpack_lane1(s15);
}

void run_one_strip_batch_16lanes(
    const float* current,
    float* next,
    hls::stream<plio_word_t>& out_j_0,
    hls::stream<plio_word_t>& out_j_1,
    hls::stream<plio_word_t>& out_j_2,
    hls::stream<plio_word_t>& out_j_3,
    hls::stream<plio_word_t>& out_j_4,
    hls::stream<plio_word_t>& out_j_5,
    hls::stream<plio_word_t>& out_j_6,
    hls::stream<plio_word_t>& out_j_7,
    hls::stream<plio_word_t>& out_j_8,
    hls::stream<plio_word_t>& out_j_9,
    hls::stream<plio_word_t>& out_j_10,
    hls::stream<plio_word_t>& out_j_11,
    hls::stream<plio_word_t>& out_j_12,
    hls::stream<plio_word_t>& out_j_13,
    hls::stream<plio_word_t>& out_j_14,
    hls::stream<plio_word_t>& out_j_15,
    hls::stream<plio_word_t>& in_j_next_0,
    hls::stream<plio_word_t>& in_j_next_1,
    hls::stream<plio_word_t>& in_j_next_2,
    hls::stream<plio_word_t>& in_j_next_3,
    hls::stream<plio_word_t>& in_j_next_4,
    hls::stream<plio_word_t>& in_j_next_5,
    hls::stream<plio_word_t>& in_j_next_6,
    hls::stream<plio_word_t>& in_j_next_7,
    hls::stream<plio_word_t>& in_j_next_8,
    hls::stream<plio_word_t>& in_j_next_9,
    hls::stream<plio_word_t>& in_j_next_10,
    hls::stream<plio_word_t>& in_j_next_11,
    hls::stream<plio_word_t>& in_j_next_12,
    hls::stream<plio_word_t>& in_j_next_13,
    hls::stream<plio_word_t>& in_j_next_14,
    hls::stream<plio_word_t>& in_j_next_15,
    int strip_batch,
    float q0sqr,
    float lane_sum[srad_cfg::kParallelLanes],
    float lane_sum2[srad_cfg::kParallelLanes]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    hls::stream<plio_word_t> lane_stat0;
    hls::stream<plio_word_t> lane_stat1;
    hls::stream<plio_word_t> lane_stat2;
    hls::stream<plio_word_t> lane_stat3;
    hls::stream<plio_word_t> lane_stat4;
    hls::stream<plio_word_t> lane_stat5;
    hls::stream<plio_word_t> lane_stat6;
    hls::stream<plio_word_t> lane_stat7;
    hls::stream<plio_word_t> lane_stat8;
    hls::stream<plio_word_t> lane_stat9;
    hls::stream<plio_word_t> lane_stat10;
    hls::stream<plio_word_t> lane_stat11;
    hls::stream<plio_word_t> lane_stat12;
    hls::stream<plio_word_t> lane_stat13;
    hls::stream<plio_word_t> lane_stat14;
    hls::stream<plio_word_t> lane_stat15;
    hls::stream<plio_word_t> to_aie_words0;
    hls::stream<plio_word_t> to_aie_words1;
    hls::stream<plio_word_t> to_aie_words2;
    hls::stream<plio_word_t> to_aie_words3;
    hls::stream<plio_word_t> to_aie_words4;
    hls::stream<plio_word_t> to_aie_words5;
    hls::stream<plio_word_t> to_aie_words6;
    hls::stream<plio_word_t> to_aie_words7;
    hls::stream<plio_word_t> to_aie_words8;
    hls::stream<plio_word_t> to_aie_words9;
    hls::stream<plio_word_t> to_aie_words10;
    hls::stream<plio_word_t> to_aie_words11;
    hls::stream<plio_word_t> to_aie_words12;
    hls::stream<plio_word_t> to_aie_words13;
    hls::stream<plio_word_t> to_aie_words14;
    hls::stream<plio_word_t> to_aie_words15;
    hls::stream<plio_word_t> from_aie_words0;
    hls::stream<plio_word_t> from_aie_words1;
    hls::stream<plio_word_t> from_aie_words2;
    hls::stream<plio_word_t> from_aie_words3;
    hls::stream<plio_word_t> from_aie_words4;
    hls::stream<plio_word_t> from_aie_words5;
    hls::stream<plio_word_t> from_aie_words6;
    hls::stream<plio_word_t> from_aie_words7;
    hls::stream<plio_word_t> from_aie_words8;
    hls::stream<plio_word_t> from_aie_words9;
    hls::stream<plio_word_t> from_aie_words10;
    hls::stream<plio_word_t> from_aie_words11;
    hls::stream<plio_word_t> from_aie_words12;
    hls::stream<plio_word_t> from_aie_words13;
    hls::stream<plio_word_t> from_aie_words14;
    hls::stream<plio_word_t> from_aie_words15;
#pragma HLS STREAM variable=lane_stat0 depth=2
#pragma HLS STREAM variable=lane_stat1 depth=2
#pragma HLS STREAM variable=lane_stat2 depth=2
#pragma HLS STREAM variable=lane_stat3 depth=2
#pragma HLS STREAM variable=lane_stat4 depth=2
#pragma HLS STREAM variable=lane_stat5 depth=2
#pragma HLS STREAM variable=lane_stat6 depth=2
#pragma HLS STREAM variable=lane_stat7 depth=2
#pragma HLS STREAM variable=lane_stat8 depth=2
#pragma HLS STREAM variable=lane_stat9 depth=2
#pragma HLS STREAM variable=lane_stat10 depth=2
#pragma HLS STREAM variable=lane_stat11 depth=2
#pragma HLS STREAM variable=lane_stat12 depth=2
#pragma HLS STREAM variable=lane_stat13 depth=2
#pragma HLS STREAM variable=lane_stat14 depth=2
#pragma HLS STREAM variable=lane_stat15 depth=2
#pragma HLS STREAM variable=to_aie_words0 depth=128
#pragma HLS STREAM variable=to_aie_words1 depth=128
#pragma HLS STREAM variable=to_aie_words2 depth=128
#pragma HLS STREAM variable=to_aie_words3 depth=128
#pragma HLS STREAM variable=to_aie_words4 depth=128
#pragma HLS STREAM variable=to_aie_words5 depth=128
#pragma HLS STREAM variable=to_aie_words6 depth=128
#pragma HLS STREAM variable=to_aie_words7 depth=128
#pragma HLS STREAM variable=to_aie_words8 depth=128
#pragma HLS STREAM variable=to_aie_words9 depth=128
#pragma HLS STREAM variable=to_aie_words10 depth=128
#pragma HLS STREAM variable=to_aie_words11 depth=128
#pragma HLS STREAM variable=to_aie_words12 depth=128
#pragma HLS STREAM variable=to_aie_words13 depth=128
#pragma HLS STREAM variable=to_aie_words14 depth=128
#pragma HLS STREAM variable=to_aie_words15 depth=128
#pragma HLS STREAM variable=from_aie_words0 depth=128
#pragma HLS STREAM variable=from_aie_words1 depth=128
#pragma HLS STREAM variable=from_aie_words2 depth=128
#pragma HLS STREAM variable=from_aie_words3 depth=128
#pragma HLS STREAM variable=from_aie_words4 depth=128
#pragma HLS STREAM variable=from_aie_words5 depth=128
#pragma HLS STREAM variable=from_aie_words6 depth=128
#pragma HLS STREAM variable=from_aie_words7 depth=128
#pragma HLS STREAM variable=from_aie_words8 depth=128
#pragma HLS STREAM variable=from_aie_words9 depth=128
#pragma HLS STREAM variable=from_aie_words10 depth=128
#pragma HLS STREAM variable=from_aie_words11 depth=128
#pragma HLS STREAM variable=from_aie_words12 depth=128
#pragma HLS STREAM variable=from_aie_words13 depth=128
#pragma HLS STREAM variable=from_aie_words14 depth=128
#pragma HLS STREAM variable=from_aie_words15 depth=128

#pragma HLS DATAFLOW
    load_16lane_input_rows(current,
                           strip_batch,
                           q0sqr,
                           to_aie_words0,
                           to_aie_words1,
                           to_aie_words2,
                           to_aie_words3,
                           to_aie_words4,
                           to_aie_words5,
                           to_aie_words6,
                           to_aie_words7,
                           to_aie_words8,
                           to_aie_words9,
                           to_aie_words10,
                           to_aie_words11,
                           to_aie_words12,
                           to_aie_words13,
                           to_aie_words14,
                           to_aie_words15);
    run_one_lane_stream(to_aie_words0, out_j_0, in_j_next_0, from_aie_words0);
    run_one_lane_stream(to_aie_words1, out_j_1, in_j_next_1, from_aie_words1);
    run_one_lane_stream(to_aie_words2, out_j_2, in_j_next_2, from_aie_words2);
    run_one_lane_stream(to_aie_words3, out_j_3, in_j_next_3, from_aie_words3);
    run_one_lane_stream(to_aie_words4, out_j_4, in_j_next_4, from_aie_words4);
    run_one_lane_stream(to_aie_words5, out_j_5, in_j_next_5, from_aie_words5);
    run_one_lane_stream(to_aie_words6, out_j_6, in_j_next_6, from_aie_words6);
    run_one_lane_stream(to_aie_words7, out_j_7, in_j_next_7, from_aie_words7);
    run_one_lane_stream(to_aie_words8, out_j_8, in_j_next_8, from_aie_words8);
    run_one_lane_stream(to_aie_words9, out_j_9, in_j_next_9, from_aie_words9);
    run_one_lane_stream(to_aie_words10, out_j_10, in_j_next_10, from_aie_words10);
    run_one_lane_stream(to_aie_words11, out_j_11, in_j_next_11, from_aie_words11);
    run_one_lane_stream(to_aie_words12, out_j_12, in_j_next_12, from_aie_words12);
    run_one_lane_stream(to_aie_words13, out_j_13, in_j_next_13, from_aie_words13);
    run_one_lane_stream(to_aie_words14, out_j_14, in_j_next_14, from_aie_words14);
    run_one_lane_stream(to_aie_words15, out_j_15, in_j_next_15, from_aie_words15);
    store_16lane_output_rows(next, strip_batch,
                             from_aie_words0,
                             from_aie_words1,
                             from_aie_words2,
                             from_aie_words3,
                             from_aie_words4,
                             from_aie_words5,
                             from_aie_words6,
                             from_aie_words7,
                             from_aie_words8,
                             from_aie_words9,
                             from_aie_words10,
                             from_aie_words11,
                             from_aie_words12,
                             from_aie_words13,
                             from_aie_words14,
                             from_aie_words15,
                             lane_stat0,
                             lane_stat1,
                             lane_stat2,
                             lane_stat3,
                             lane_stat4,
                             lane_stat5,
                             lane_stat6,
                             lane_stat7,
                             lane_stat8,
                             lane_stat9,
                             lane_stat10,
                             lane_stat11,
                             lane_stat12,
                             lane_stat13,
                             lane_stat14,
                             lane_stat15);
    collect_16lane_stats(lane_stat0,
                         lane_stat1,
                         lane_stat2,
                         lane_stat3,
                         lane_stat4,
                         lane_stat5,
                         lane_stat6,
                         lane_stat7,
                         lane_stat8,
                         lane_stat9,
                         lane_stat10,
                         lane_stat11,
                         lane_stat12,
                         lane_stat13,
                         lane_stat14,
                         lane_stat15,
                         lane_sum,
                         lane_sum2);
}

void run_all_strips(const float* current,
                    float* next,
                    hls::stream<plio_word_t> out_j[srad_cfg::kParallelLanes],
                    hls::stream<plio_word_t> in_j_next[srad_cfg::kParallelLanes],
                    float q0sqr,
                    float& next_sum,
                    float& next_sum2) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
#pragma HLS ALLOCATION instances=run_one_strip_batch_16lanes limit=1 function
    next_sum = 0.0f;
    next_sum2 = 0.0f;

    for (int strip_batch = 0;
         strip_batch < kStripBatches;
         ++strip_batch) {
        float lane_sum[srad_cfg::kParallelLanes];
        float lane_sum2[srad_cfg::kParallelLanes];
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

        run_one_strip_batch_16lanes(current,
                                    next,
                                    out_j[0],
                                    out_j[1],
                                    out_j[2],
                                    out_j[3],
                                    out_j[4],
                                    out_j[5],
                                    out_j[6],
                                    out_j[7],
                                    out_j[8],
                                    out_j[9],
                                    out_j[10],
                                    out_j[11],
                                    out_j[12],
                                    out_j[13],
                                    out_j[14],
                                    out_j[15],
                                    in_j_next[0],
                                    in_j_next[1],
                                    in_j_next[2],
                                    in_j_next[3],
                                    in_j_next[4],
                                    in_j_next[5],
                                    in_j_next[6],
                                    in_j_next[7],
                                    in_j_next[8],
                                    in_j_next[9],
                                    in_j_next[10],
                                    in_j_next[11],
                                    in_j_next[12],
                                    in_j_next[13],
                                    in_j_next[14],
                                    in_j_next[15],
                                    strip_batch,
                                    q0sqr,
                                    lane_sum,
                                    lane_sum2);

        for (int lane = 0; lane < srad_cfg::kParallelLanes; ++lane) {
#pragma HLS UNROLL
            next_sum += lane_sum[lane];
            next_sum2 += lane_sum2[lane];
        }
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
           hls::stream<plio_word_t> out_j[srad_cfg::kParallelLanes],
           hls::stream<plio_word_t> in_j_next[srad_cfg::kParallelLanes]) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 \
    max_read_burst_length=64 max_write_burst_length=64 \
    num_read_outstanding=16 num_write_outstanding=16 \
    max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 \
    max_read_burst_length=64 max_write_burst_length=64 \
    num_read_outstanding=16 num_write_outstanding=16 \
    max_widen_bitwidth=512
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
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
