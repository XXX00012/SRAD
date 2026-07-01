#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kTopPlWorkers == 1,
              "ours_32lane board path expects a single TopPL CU");
static_assert(srad_cfg::kParallelLanes == 32,
              "ours_32lane board path expects thirty-two AIE row-stream lanes");
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

constexpr int kLaneCount = srad_cfg::kParallelLanes;
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
constexpr int kOutputCaptureGroupLanes = 16;

static_assert(kLaneCount == 32,
              "TopPL column batch path expects thirty-two lanes");
static_assert(kLanesPerBatch == 32,
              "TopPL column batch path expects thirty-two lanes per batch");
static_assert(kStripBatches == 1,
              "4000 columns with 32 lanes of 125 columns should use one batch");
static_assert((kLaneCount % kOutputCaptureGroupLanes) == 0,
              "output capture groups must exactly cover all lanes");
static_assert((kLaneCount / kOutputCaptureGroupLanes) == 2,
              "output capture is split into two sixteen-lane groups");
static_assert(kBatchDataElems == srad_cfg::kBoardCols,
              "one 32-lane batch should cover a full board row");
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

float buffered_input_value(float row_elems[kBatchDataElems],
                           int lane,
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
        return row_elems[lane * srad_cfg::kRowDataElems + physical_col];
    }
    return 0.0f;
}

void load_lane_input_rows(const float* current,
                          int strip_batch,
                          float q0sqr,
                          hls::stream<plio_word_t> to_aie_words[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=to_aie_words complete dim=1
    float row_elems[kBatchDataElems];
#pragma HLS ARRAY_PARTITION variable=row_elems cyclic factor=32 dim=1

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

            for (int lane = 0; lane < kLaneCount; ++lane) {
#pragma HLS UNROLL
                to_aie_words[lane].write(pack_two_floats(
                    buffered_input_value(row_elems, lane, col0, valid_row, q0sqr),
                    buffered_input_value(row_elems, lane, col1, valid_row, q0sqr)));
            }
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

void capture_lane_output_word(int lane,
                              int word_col,
                              hls::stream<plio_word_t>& from_aie_words,
                              float row_elems[kBatchDataElems],
                              float& lane_sum,
                              float& lane_sum2) {
#pragma HLS INLINE
    const plio_word_t word = from_aie_words.read();
    const int physical_col = word_col * 2;
    const int next_physical_col = physical_col + 1;
    const int lane_base = lane * srad_cfg::kRowDataElems;
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

template <int LaneBase>
void capture_lane_output_group(
    int word_col,
    hls::stream<plio_word_t> from_aie_words[kLaneCount],
    float row_elems[kBatchDataElems],
    float lane_sum[kLaneCount],
    float lane_sum2[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1
    for (int lane_offset = 0;
         lane_offset < kOutputCaptureGroupLanes;
         ++lane_offset) {
#pragma HLS UNROLL
        const int lane = LaneBase + lane_offset;
        capture_lane_output_word(lane, word_col, from_aie_words[lane],
                                 row_elems, lane_sum[lane], lane_sum2[lane]);
    }
}

void capture_lane_output_row(hls::stream<plio_word_t> from_aie_words[kLaneCount],
                             float row_elems[kBatchDataElems],
                             float lane_sum[kLaneCount],
                             float lane_sum2[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
        capture_lane_output_group<0>(word_col,
                                     from_aie_words,
                                     row_elems,
                                     lane_sum,
                                     lane_sum2);
        capture_lane_output_group<kOutputCaptureGroupLanes>(word_col,
                                                            from_aie_words,
                                                            row_elems,
                                                            lane_sum,
                                                            lane_sum2);
    }
}

template <int LaneBase>
void discard_lane_output_group(
    hls::stream<plio_word_t> from_aie_words[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
    for (int lane_offset = 0;
         lane_offset < kOutputCaptureGroupLanes;
         ++lane_offset) {
#pragma HLS UNROLL
        (void)from_aie_words[LaneBase + lane_offset].read();
    }
}

void discard_lane_output_row(hls::stream<plio_word_t> from_aie_words[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
        discard_lane_output_group<0>(from_aie_words);
        discard_lane_output_group<kOutputCaptureGroupLanes>(from_aie_words);
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

void store_lane_output_rows(float* next,
                            int strip_batch,
                            hls::stream<plio_word_t> from_aie_words[kLaneCount],
                            hls::stream<plio_word_t> lane_stats[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_stats complete dim=1
    float row_elems[kBatchDataElems];
    float lane_sum[kLaneCount];
    float lane_sum2[kLaneCount];
#pragma HLS ARRAY_PARTITION variable=row_elems cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    for (int lane = 0; lane < kLaneCount; ++lane) {
#pragma HLS UNROLL
        lane_sum[lane] = 0.0f;
        lane_sum2[lane] = 0.0f;
    }

    for (int stream_row = 0;
         stream_row < kLeadingInvalidOutputRows;
         ++stream_row) {
        discard_lane_output_row(from_aie_words);
    }

    for (int board_row = 0;
         board_row < srad_cfg::kBoardRows;
         ++board_row) {
        capture_lane_output_row(from_aie_words, row_elems, lane_sum, lane_sum2);
        write_contiguous_output_row(next, strip_batch, board_row, row_elems);
    }

    for (int lane = 0; lane < kLaneCount; ++lane) {
#pragma HLS UNROLL
        lane_stats[lane].write(pack_two_floats(lane_sum[lane], lane_sum2[lane]));
    }
}

void collect_lane_stats(hls::stream<plio_word_t> lane_stats[kLaneCount],
                        float lane_sum[kLaneCount],
                        float lane_sum2[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_stats complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    for (int lane = 0; lane < kLaneCount; ++lane) {
#pragma HLS UNROLL
        const plio_word_t stat = lane_stats[lane].read();
        lane_sum[lane] = unpack_lane0(stat);
        lane_sum2[lane] = unpack_lane1(stat);
    }
}

void run_one_strip_batch_lanes(
    const float* current,
    float* next,
    hls::stream<plio_word_t> out_j[kLaneCount],
    hls::stream<plio_word_t> in_j_next[kLaneCount],
    int strip_batch,
    float q0sqr,
    float lane_sum[kLaneCount],
    float lane_sum2[kLaneCount]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    hls::stream<plio_word_t> lane_stats[kLaneCount];
    hls::stream<plio_word_t> to_aie_words[kLaneCount];
    hls::stream<plio_word_t> from_aie_words[kLaneCount];
#pragma HLS ARRAY_PARTITION variable=lane_stats complete dim=1
#pragma HLS ARRAY_PARTITION variable=to_aie_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=from_aie_words complete dim=1
#pragma HLS STREAM variable=lane_stats depth=2
#pragma HLS STREAM variable=to_aie_words depth=128
#pragma HLS STREAM variable=from_aie_words depth=128

#pragma HLS DATAFLOW
    load_lane_input_rows(current, strip_batch, q0sqr, to_aie_words);
#define RUN_LANE_STREAM(N) \
    run_one_lane_stream(to_aie_words[N], out_j[N], in_j_next[N], from_aie_words[N])
    RUN_LANE_STREAM(0);
    RUN_LANE_STREAM(1);
    RUN_LANE_STREAM(2);
    RUN_LANE_STREAM(3);
    RUN_LANE_STREAM(4);
    RUN_LANE_STREAM(5);
    RUN_LANE_STREAM(6);
    RUN_LANE_STREAM(7);
    RUN_LANE_STREAM(8);
    RUN_LANE_STREAM(9);
    RUN_LANE_STREAM(10);
    RUN_LANE_STREAM(11);
    RUN_LANE_STREAM(12);
    RUN_LANE_STREAM(13);
    RUN_LANE_STREAM(14);
    RUN_LANE_STREAM(15);
    RUN_LANE_STREAM(16);
    RUN_LANE_STREAM(17);
    RUN_LANE_STREAM(18);
    RUN_LANE_STREAM(19);
    RUN_LANE_STREAM(20);
    RUN_LANE_STREAM(21);
    RUN_LANE_STREAM(22);
    RUN_LANE_STREAM(23);
    RUN_LANE_STREAM(24);
    RUN_LANE_STREAM(25);
    RUN_LANE_STREAM(26);
    RUN_LANE_STREAM(27);
    RUN_LANE_STREAM(28);
    RUN_LANE_STREAM(29);
    RUN_LANE_STREAM(30);
    RUN_LANE_STREAM(31);
#undef RUN_LANE_STREAM
    store_lane_output_rows(next, strip_batch, from_aie_words, lane_stats);
    collect_lane_stats(lane_stats, lane_sum, lane_sum2);
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
#pragma HLS ALLOCATION instances=run_one_strip_batch_lanes limit=1 function
    next_sum = 0.0f;
    next_sum2 = 0.0f;

    for (int strip_batch = 0;
         strip_batch < kStripBatches;
         ++strip_batch) {
        float lane_sum[kLaneCount];
        float lane_sum2[kLaneCount];
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

        run_one_strip_batch_lanes(current,
                                  next,
                                  out_j,
                                  in_j_next,
                                  strip_batch,
                                  q0sqr,
                                  lane_sum,
                                  lane_sum2);

        for (int lane = 0; lane < kLaneCount; ++lane) {
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
