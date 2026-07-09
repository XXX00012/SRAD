#include "../aie/Config.h"
#include "../packet_ids_c.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;
using packet_word_t = ap_uint<32>;

constexpr int kTopPlDebugSlots = 16;
constexpr int kTopPlDebugBase = 0;

static_assert(srad_cfg::kTopPlWorkers == 12,
              "ours_192lane board path expects twelve TopPL CUs");
static_assert(srad_cfg::kTopPlColumnWorkers == 2,
              "ours_192lane board path expects two column workers per row block");
static_assert(srad_cfg::kBoardRowBlocks == 6,
              "ours_192lane board path expects six row blocks");
static_assert(srad_cfg::kLanesPerTopPl == 16,
              "each TopPL CU expects sixteen AIE row-stream lanes");
static_assert(srad_cfg::kMergedOutputsPerTopPl == 4,
              "each TopPL CU expects four merged AIE output streams");
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
constexpr int kLanesPerWorker = srad_cfg::kLanesPerTopPl;
constexpr int kLeadingInvalidOutputRows =
    srad_cfg::kBoardLanePreContextRows + srad_cfg::kCenterRowLag;
constexpr int kFullDataOutputWords = srad_cfg::kRowDataElems / 2;
constexpr int kLastDataPhysicalCol = kFullDataOutputWords * 2;
constexpr int kSum2PhysicalCol = kLastDataPhysicalCol + 2;
constexpr int kWorkerDataElems =
    kLanesPerWorker * srad_cfg::kRowDataElems;

static_assert(kLanesPerWorker == 16,
              "TopPL worker path expects sixteen lanes per CU");
static_assert(kWorkerDataElems == srad_cfg::kWorkerCols,
              "TopPL worker should read one contiguous half-image row");
static_assert(srad_cfg::kBoardStripBatches == 1,
              "each 32-lane row block should cover all board strips in one graph batch");
static_assert(kStreamRowsPerLane ==
                  kLeadingInvalidOutputRows + srad_cfg::kRowsPerRowBlock,
              "output stream should contain one padded row block plus leading invalid rows");
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

int worker_col_id(int worker_id) {
#pragma HLS INLINE
    return clamp_worker_id(worker_id) % srad_cfg::kTopPlColumnWorkers;
}

int worker_row_block_id(int worker_id) {
#pragma HLS INLINE
    return clamp_worker_id(worker_id) / srad_cfg::kTopPlColumnWorkers;
}

int worker_col_base(int worker_id) {
#pragma HLS INLINE
    return worker_col_id(worker_id) * srad_cfg::kWorkerCols;
}

int worker_row_base(int worker_id) {
#pragma HLS INLINE
    return worker_row_block_id(worker_id) * srad_cfg::kRowsPerRowBlock;
}

bool real_board_row(int board_row) {
#pragma HLS INLINE
    return (board_row >= 0) && (board_row < srad_cfg::kBoardRows);
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

void compute_initial_worker_stats(const float* image,
                                  int worker_id,
                                  float& sum,
                                  float& sum2) {
#pragma HLS INLINE off
    sum = 0.0f;
    sum2 = 0.0f;
    const int col_base = worker_col_base(worker_id);
    const int row_base_global = worker_row_base(worker_id);

    for (int local_row = 0;
         local_row < srad_cfg::kRowsPerRowBlock;
         ++local_row) {
        const int board_row = row_base_global + local_row;
        if (real_board_row(board_row)) {
            const int row_base = board_row * srad_cfg::kBoardCols + col_base;
            for (int elem = 0; elem < kWorkerDataElems; ++elem) {
#pragma HLS PIPELINE II=1
                const float value = image[row_base + elem];
                sum += value;
                sum2 += value * value;
            }
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

void write_toppl_debug(float* debug, int worker_id, int slot, float value) {
#pragma HLS INLINE
    debug[kTopPlDebugBase + worker_id * kTopPlDebugSlots + slot] = value;
}

packet_word_t read_packet_word32(hls::stream<plio_word_t>& in_j_next,
                                 plio_word_t& cached_word,
                                 bool& use_cached_high) {
#pragma HLS INLINE
    packet_word_t word = 0;
    if (use_cached_high) {
        word = cached_word.range(63, 32);
        use_cached_high = false;
    } else {
        cached_word = in_j_next.read();
        word = cached_word.range(31, 0);
        use_cached_high = true;
    }
    return word;
}

int packet_id_from_header(packet_word_t header) {
#pragma HLS INLINE
    return static_cast<int>(header.range(4, 0));
}

int packet_merge_way(int global_merge_id, int packet_id) {
#pragma HLS INLINE
    static const int kPacketIds[srad_cfg::kMergedOutputPlioCount]
                               [srad_cfg::kOutputMergeWays] = {
    {ours_plq0_out_j_next_merged_0_0, ours_plq0_out_j_next_merged_0_1, ours_plq0_out_j_next_merged_0_2, ours_plq0_out_j_next_merged_0_3},
    {ours_plq0_out_j_next_merged_1_0, ours_plq0_out_j_next_merged_1_1, ours_plq0_out_j_next_merged_1_2, ours_plq0_out_j_next_merged_1_3},
    {ours_plq0_out_j_next_merged_2_0, ours_plq0_out_j_next_merged_2_1, ours_plq0_out_j_next_merged_2_2, ours_plq0_out_j_next_merged_2_3},
    {ours_plq0_out_j_next_merged_3_0, ours_plq0_out_j_next_merged_3_1, ours_plq0_out_j_next_merged_3_2, ours_plq0_out_j_next_merged_3_3},
    {ours_plq0_out_j_next_merged_4_0, ours_plq0_out_j_next_merged_4_1, ours_plq0_out_j_next_merged_4_2, ours_plq0_out_j_next_merged_4_3},
    {ours_plq0_out_j_next_merged_5_0, ours_plq0_out_j_next_merged_5_1, ours_plq0_out_j_next_merged_5_2, ours_plq0_out_j_next_merged_5_3},
    {ours_plq0_out_j_next_merged_6_0, ours_plq0_out_j_next_merged_6_1, ours_plq0_out_j_next_merged_6_2, ours_plq0_out_j_next_merged_6_3},
    {ours_plq0_out_j_next_merged_7_0, ours_plq0_out_j_next_merged_7_1, ours_plq0_out_j_next_merged_7_2, ours_plq0_out_j_next_merged_7_3},
    {ours_plq0_out_j_next_merged_8_0, ours_plq0_out_j_next_merged_8_1, ours_plq0_out_j_next_merged_8_2, ours_plq0_out_j_next_merged_8_3},
    {ours_plq0_out_j_next_merged_9_0, ours_plq0_out_j_next_merged_9_1, ours_plq0_out_j_next_merged_9_2, ours_plq0_out_j_next_merged_9_3},
    {ours_plq0_out_j_next_merged_10_0, ours_plq0_out_j_next_merged_10_1, ours_plq0_out_j_next_merged_10_2, ours_plq0_out_j_next_merged_10_3},
    {ours_plq0_out_j_next_merged_11_0, ours_plq0_out_j_next_merged_11_1, ours_plq0_out_j_next_merged_11_2, ours_plq0_out_j_next_merged_11_3},
    {ours_plq0_out_j_next_merged_12_0, ours_plq0_out_j_next_merged_12_1, ours_plq0_out_j_next_merged_12_2, ours_plq0_out_j_next_merged_12_3},
    {ours_plq0_out_j_next_merged_13_0, ours_plq0_out_j_next_merged_13_1, ours_plq0_out_j_next_merged_13_2, ours_plq0_out_j_next_merged_13_3},
    {ours_plq0_out_j_next_merged_14_0, ours_plq0_out_j_next_merged_14_1, ours_plq0_out_j_next_merged_14_2, ours_plq0_out_j_next_merged_14_3},
    {ours_plq0_out_j_next_merged_15_0, ours_plq0_out_j_next_merged_15_1, ours_plq0_out_j_next_merged_15_2, ours_plq0_out_j_next_merged_15_3},
    {ours_plq0_out_j_next_merged_16_0, ours_plq0_out_j_next_merged_16_1, ours_plq0_out_j_next_merged_16_2, ours_plq0_out_j_next_merged_16_3},
    {ours_plq0_out_j_next_merged_17_0, ours_plq0_out_j_next_merged_17_1, ours_plq0_out_j_next_merged_17_2, ours_plq0_out_j_next_merged_17_3},
    {ours_plq0_out_j_next_merged_18_0, ours_plq0_out_j_next_merged_18_1, ours_plq0_out_j_next_merged_18_2, ours_plq0_out_j_next_merged_18_3},
    {ours_plq0_out_j_next_merged_19_0, ours_plq0_out_j_next_merged_19_1, ours_plq0_out_j_next_merged_19_2, ours_plq0_out_j_next_merged_19_3},
    {ours_plq0_out_j_next_merged_20_0, ours_plq0_out_j_next_merged_20_1, ours_plq0_out_j_next_merged_20_2, ours_plq0_out_j_next_merged_20_3},
    {ours_plq0_out_j_next_merged_21_0, ours_plq0_out_j_next_merged_21_1, ours_plq0_out_j_next_merged_21_2, ours_plq0_out_j_next_merged_21_3},
    {ours_plq0_out_j_next_merged_22_0, ours_plq0_out_j_next_merged_22_1, ours_plq0_out_j_next_merged_22_2, ours_plq0_out_j_next_merged_22_3},
    {ours_plq0_out_j_next_merged_23_0, ours_plq0_out_j_next_merged_23_1, ours_plq0_out_j_next_merged_23_2, ours_plq0_out_j_next_merged_23_3},
    {ours_plq0_out_j_next_merged_24_0, ours_plq0_out_j_next_merged_24_1, ours_plq0_out_j_next_merged_24_2, ours_plq0_out_j_next_merged_24_3},
    {ours_plq0_out_j_next_merged_25_0, ours_plq0_out_j_next_merged_25_1, ours_plq0_out_j_next_merged_25_2, ours_plq0_out_j_next_merged_25_3},
    {ours_plq0_out_j_next_merged_26_0, ours_plq0_out_j_next_merged_26_1, ours_plq0_out_j_next_merged_26_2, ours_plq0_out_j_next_merged_26_3},
    {ours_plq0_out_j_next_merged_27_0, ours_plq0_out_j_next_merged_27_1, ours_plq0_out_j_next_merged_27_2, ours_plq0_out_j_next_merged_27_3},
    {ours_plq0_out_j_next_merged_28_0, ours_plq0_out_j_next_merged_28_1, ours_plq0_out_j_next_merged_28_2, ours_plq0_out_j_next_merged_28_3},
    {ours_plq0_out_j_next_merged_29_0, ours_plq0_out_j_next_merged_29_1, ours_plq0_out_j_next_merged_29_2, ours_plq0_out_j_next_merged_29_3},
    {ours_plq0_out_j_next_merged_30_0, ours_plq0_out_j_next_merged_30_1, ours_plq0_out_j_next_merged_30_2, ours_plq0_out_j_next_merged_30_3},
    {ours_plq0_out_j_next_merged_31_0, ours_plq0_out_j_next_merged_31_1, ours_plq0_out_j_next_merged_31_2, ours_plq0_out_j_next_merged_31_3},
    {ours_plq0_out_j_next_merged_32_0, ours_plq0_out_j_next_merged_32_1, ours_plq0_out_j_next_merged_32_2, ours_plq0_out_j_next_merged_32_3},
    {ours_plq0_out_j_next_merged_33_0, ours_plq0_out_j_next_merged_33_1, ours_plq0_out_j_next_merged_33_2, ours_plq0_out_j_next_merged_33_3},
    {ours_plq0_out_j_next_merged_34_0, ours_plq0_out_j_next_merged_34_1, ours_plq0_out_j_next_merged_34_2, ours_plq0_out_j_next_merged_34_3},
    {ours_plq0_out_j_next_merged_35_0, ours_plq0_out_j_next_merged_35_1, ours_plq0_out_j_next_merged_35_2, ours_plq0_out_j_next_merged_35_3},
    {ours_plq0_out_j_next_merged_36_0, ours_plq0_out_j_next_merged_36_1, ours_plq0_out_j_next_merged_36_2, ours_plq0_out_j_next_merged_36_3},
    {ours_plq0_out_j_next_merged_37_0, ours_plq0_out_j_next_merged_37_1, ours_plq0_out_j_next_merged_37_2, ours_plq0_out_j_next_merged_37_3},
    {ours_plq0_out_j_next_merged_38_0, ours_plq0_out_j_next_merged_38_1, ours_plq0_out_j_next_merged_38_2, ours_plq0_out_j_next_merged_38_3},
    {ours_plq0_out_j_next_merged_39_0, ours_plq0_out_j_next_merged_39_1, ours_plq0_out_j_next_merged_39_2, ours_plq0_out_j_next_merged_39_3},
    {ours_plq0_out_j_next_merged_40_0, ours_plq0_out_j_next_merged_40_1, ours_plq0_out_j_next_merged_40_2, ours_plq0_out_j_next_merged_40_3},
    {ours_plq0_out_j_next_merged_41_0, ours_plq0_out_j_next_merged_41_1, ours_plq0_out_j_next_merged_41_2, ours_plq0_out_j_next_merged_41_3},
    {ours_plq0_out_j_next_merged_42_0, ours_plq0_out_j_next_merged_42_1, ours_plq0_out_j_next_merged_42_2, ours_plq0_out_j_next_merged_42_3},
    {ours_plq0_out_j_next_merged_43_0, ours_plq0_out_j_next_merged_43_1, ours_plq0_out_j_next_merged_43_2, ours_plq0_out_j_next_merged_43_3},
    {ours_plq0_out_j_next_merged_44_0, ours_plq0_out_j_next_merged_44_1, ours_plq0_out_j_next_merged_44_2, ours_plq0_out_j_next_merged_44_3},
    {ours_plq0_out_j_next_merged_45_0, ours_plq0_out_j_next_merged_45_1, ours_plq0_out_j_next_merged_45_2, ours_plq0_out_j_next_merged_45_3},
    {ours_plq0_out_j_next_merged_46_0, ours_plq0_out_j_next_merged_46_1, ours_plq0_out_j_next_merged_46_2, ours_plq0_out_j_next_merged_46_3},
    {ours_plq0_out_j_next_merged_47_0, ours_plq0_out_j_next_merged_47_1, ours_plq0_out_j_next_merged_47_2, ours_plq0_out_j_next_merged_47_3}
    };

    int merge = global_merge_id;
    if (merge < 0) {
        merge = 0;
    }
    if (merge >= srad_cfg::kMergedOutputPlioCount) {
        merge = srad_cfg::kMergedOutputPlioCount - 1;
    }

    if (packet_id == kPacketIds[merge][0]) {
        return 0;
    }
    if (packet_id == kPacketIds[merge][1]) {
        return 1;
    }
    if (packet_id == kPacketIds[merge][2]) {
        return 2;
    }
    return 3;
}

template<int Way>
void capture_merged_payload_words(hls::stream<plio_word_t>& in_j_next,
                                  hls::stream<plio_word_t>& from_aie_words,
                                  plio_word_t& cached_word,
                                  bool& use_cached_high) {
#pragma HLS INLINE off
    for (int word = 0; word < kWordsPerRow; ++word) {
#pragma HLS PIPELINE II=1
        const packet_word_t payload0 =
            read_packet_word32(in_j_next, cached_word, use_cached_high);
        const packet_word_t payload1 =
            read_packet_word32(in_j_next, cached_word, use_cached_high);
        plio_word_t payload = 0;
        payload.range(31, 0) = payload0;
        payload.range(63, 32) = payload1;
        from_aie_words.write(payload);
    }

    // Each AIE packet contains 1 header + 128 payload 32-bit words: an odd
    // number of packet words. With 64-bit PLIO the final high half is padding,
    // so do not treat the cached high half as the next packet header.
    use_cached_high = false;
}

// Group selects which contiguous 4-lane slice of from_aie_words this merged
// PLIO feeds; the AIE-assigned way ID (runtime) then picks one of those 4
// lanes, so the lane a packet lands on is only known at compile time up to
// which group, not which of the 4 ways within it.
template<int Group>
void capture_merged_output_packets(hls::stream<plio_word_t>& in_j_next,
                                   hls::stream<plio_word_t> from_aie_words[kLanesPerWorker],
                                   int worker_id) {
#pragma HLS INLINE off
    const int global_merge_id = worker_id * srad_cfg::kMergedOutputsPerTopPl + Group;
    for (int stream_row = 0; stream_row < kStreamRowsPerLane; ++stream_row) {
        for (int packet = 0; packet < srad_cfg::kOutputMergeWays; ++packet) {
            plio_word_t cached_word = 0;
            bool use_cached_high = false;
            const packet_word_t header =
                read_packet_word32(in_j_next, cached_word, use_cached_high);
            const int packet_id = packet_id_from_header(header);
            const int way = packet_merge_way(global_merge_id, packet_id);
            if (way == 0) {
                capture_merged_payload_words<0>(in_j_next, from_aie_words[4 * Group + 0],
                                                cached_word, use_cached_high);
            } else if (way == 1) {
                capture_merged_payload_words<1>(in_j_next, from_aie_words[4 * Group + 1],
                                                cached_word, use_cached_high);
            } else if (way == 2) {
                capture_merged_payload_words<2>(in_j_next, from_aie_words[4 * Group + 2],
                                                cached_word, use_cached_high);
            } else {
                capture_merged_payload_words<3>(in_j_next, from_aie_words[4 * Group + 3],
                                                cached_word, use_cached_high);
            }
        }
    }
}

void load_16lane_input_rows(const float* current,
                            int worker_id,
                            float q0sqr,
                            hls::stream<plio_word_t> to_aie_words[kLanesPerWorker]) {
#pragma HLS INLINE off
    float row_elems[kLanesPerWorker][srad_cfg::kRowDataElems];
#pragma HLS ARRAY_PARTITION variable=row_elems cyclic factor=4 dim=1
    const int col_base = worker_col_base(worker_id);
    const int row_base_global = worker_row_base(worker_id);

    for (int stream_row = 0;
         stream_row < kStreamRowsPerLane;
         ++stream_row) {
        const int first_row =
            row_base_global - srad_cfg::kBoardLanePreContextRows;
        const int board_row = first_row + stream_row;
        const bool valid_row = real_board_row(board_row);

        if (valid_row) {
            const int row_base =
                board_row * srad_cfg::kBoardCols + col_base;

            for (int elem = 0; elem < kWorkerDataElems; ++elem) {
#pragma HLS PIPELINE II=1
                row_elems[elem / srad_cfg::kRowDataElems]
                         [elem % srad_cfg::kRowDataElems] =
                    current[row_base + elem];
            }

            for (int word_col = 0; word_col < kFullDataOutputWords; ++word_col) {
#pragma HLS PIPELINE II=1
                const int col0 = word_col * 2;
                const int col1 = col0 + 1;
                for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
                    to_aie_words[lane].write(pack_two_floats(
                        row_elems[lane][col0],
                        row_elems[lane][col1]));
                }
            }

            for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
                to_aie_words[lane].write(pack_two_floats(
                    row_elems[lane][kLastDataPhysicalCol],
                    q0sqr));
                to_aie_words[lane].write(pack_two_floats(0.0f, 0.0f));
            }
        } else {
            for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
                for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
                    to_aie_words[lane].write(pack_two_floats(0.0f, 0.0f));
                }
            }
        }
    }
}

void capture_lane_output_word(int lane,
                              int word_col,
                              hls::stream<plio_word_t>& from_aie_words,
                              float row_elems[kLanesPerWorker][srad_cfg::kRowDataElems],
                              float& row_sum,
                              float& row_sum2) {
#pragma HLS INLINE
    const plio_word_t word = from_aie_words.read();
    const int physical_col = word_col * 2;
    const int next_physical_col = physical_col + 1;
    const float v0 = unpack_lane0(word);
    const float v1 = unpack_lane1(word);

    if (physical_col < srad_cfg::kRowDataElems) {
        row_elems[lane][physical_col] = v0;
    }
    if (next_physical_col < srad_cfg::kRowDataElems) {
        row_elems[lane][next_physical_col] = v1;
    }

    if (physical_col == srad_cfg::kStatSumPadIndex - 1) {
        row_sum = v1;
    }
    if (physical_col == srad_cfg::kStatSum2PadIndex) {
        row_sum2 = v0;
    }
}

void capture_16lane_output_row(hls::stream<plio_word_t> from_aie_words[kLanesPerWorker],
                               float row_elems[kLanesPerWorker][srad_cfg::kRowDataElems],
                               float row_sum[srad_cfg::kLanesPerTopPl],
                               float row_sum2[srad_cfg::kLanesPerTopPl]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum2 complete dim=1
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
            capture_lane_output_word(lane, word_col, from_aie_words[lane],
                                     row_elems, row_sum[lane], row_sum2[lane]);
        }
    }
}

void discard_16lane_output_row(hls::stream<plio_word_t> from_aie_words[kLanesPerWorker]) {
#pragma HLS INLINE off
    for (int word_col = 0; word_col < kWordsPerRow; ++word_col) {
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
            (void)from_aie_words[lane].read();
        }
    }
}

void write_contiguous_output_row(float* next,
                                 int worker_id,
                                 int board_row,
                                 float row_elems[kLanesPerWorker][srad_cfg::kRowDataElems]) {
#pragma HLS INLINE off
    const int col_base = worker_col_base(worker_id);
    const int row_base =
        board_row * srad_cfg::kBoardCols + col_base;

    for (int elem = 0; elem < kWorkerDataElems; ++elem) {
#pragma HLS PIPELINE II=1
        next[row_base + elem] =
            row_elems[elem / srad_cfg::kRowDataElems]
                     [elem % srad_cfg::kRowDataElems];
    }
}

void accumulate_16lane_row_stats(float lane_sum[srad_cfg::kLanesPerTopPl],
                                 float lane_sum2[srad_cfg::kLanesPerTopPl],
                                 float row_sum[srad_cfg::kLanesPerTopPl],
                                 float row_sum2[srad_cfg::kLanesPerTopPl]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum2 complete dim=1

    for (int lane = 0; lane < srad_cfg::kLanesPerTopPl; ++lane) {
#pragma HLS PIPELINE off
        float next_sum;
#pragma HLS BIND_OP variable=next_sum op=fadd impl=fulldsp latency=12
        next_sum = lane_sum[lane] + row_sum[lane];

        float next_sum2;
#pragma HLS BIND_OP variable=next_sum2 op=fadd impl=fulldsp latency=12
        next_sum2 = lane_sum2[lane] + row_sum2[lane];

        lane_sum[lane] = next_sum;
        lane_sum2[lane] = next_sum2;
    }
}

void store_16lane_output_rows(float* next,
                              int worker_id,
                              hls::stream<plio_word_t> from_aie_words[kLanesPerWorker],
                              hls::stream<plio_word_t> lane_stat[kLanesPerWorker]) {
#pragma HLS INLINE off
    float row_elems[kLanesPerWorker][srad_cfg::kRowDataElems];
    float row_sum[srad_cfg::kLanesPerTopPl];
    float row_sum2[srad_cfg::kLanesPerTopPl];
    float lane_sum[srad_cfg::kLanesPerTopPl];
    float lane_sum2[srad_cfg::kLanesPerTopPl];
#pragma HLS ARRAY_PARTITION variable=row_elems complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_sum2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    for (int lane = 0; lane < srad_cfg::kLanesPerTopPl; ++lane) {
#pragma HLS UNROLL
        lane_sum[lane] = 0.0f;
        lane_sum2[lane] = 0.0f;
    }

    for (int stream_row = 0;
         stream_row < kLeadingInvalidOutputRows;
         ++stream_row) {
        discard_16lane_output_row(from_aie_words);
    }

    const int row_base_global = worker_row_base(worker_id);
    for (int local_row = 0;
         local_row < srad_cfg::kRowsPerRowBlock;
         ++local_row) {
        const int board_row = row_base_global + local_row;
        for (int lane = 0; lane < srad_cfg::kLanesPerTopPl; ++lane) {
            row_sum[lane] = 0.0f;
            row_sum2[lane] = 0.0f;
        }

        capture_16lane_output_row(from_aie_words, row_elems, row_sum, row_sum2);

        if (real_board_row(board_row)) {
            accumulate_16lane_row_stats(lane_sum, lane_sum2,
                                        row_sum, row_sum2);

            write_contiguous_output_row(next, worker_id, board_row, row_elems);
        }
    }

    for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
        lane_stat[lane].write(pack_two_floats(lane_sum[lane], lane_sum2[lane]));
    }
}

void collect_16lane_stats(hls::stream<plio_word_t> lane_stat[kLanesPerWorker],
                          float lane_sum[srad_cfg::kLanesPerTopPl],
                          float lane_sum2[srad_cfg::kLanesPerTopPl]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS PIPELINE II=1
        const plio_word_t s = lane_stat[lane].read();
        lane_sum[lane] = unpack_lane0(s);
        lane_sum2[lane] = unpack_lane1(s);
    }
}

void run_one_strip_batch_16lanes(
    const float* current,
    float* next,
    hls::stream<plio_word_t> out_j[kLanesPerWorker],
    hls::stream<plio_word_t> in_j_next[srad_cfg::kMergedOutputsPerTopPl],
    int worker_id,
    float q0sqr,
    float lane_sum[srad_cfg::kLanesPerTopPl],
    float lane_sum2[srad_cfg::kLanesPerTopPl]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    hls::stream<plio_word_t> lane_stat[kLanesPerWorker];
    hls::stream<plio_word_t> to_aie_words[kLanesPerWorker];
    hls::stream<plio_word_t> from_aie_words[kLanesPerWorker];
#pragma HLS STREAM variable=lane_stat depth=1
#pragma HLS STREAM variable=to_aie_words depth=16
#pragma HLS bind_storage variable=to_aie_words type=fifo impl=srl
#pragma HLS bind_storage variable=from_aie_words type=fifo impl=lutram
#pragma HLS bind_storage variable=lane_stat type=fifo impl=srl
#pragma HLS STREAM variable=from_aie_words depth=64

#pragma HLS DATAFLOW disable_start_propagation
    load_16lane_input_rows(current, worker_id, q0sqr, to_aie_words);
    for (int lane = 0; lane < kLanesPerWorker; ++lane) {
#pragma HLS UNROLL
        forward_input_words(to_aie_words[lane], out_j[lane]);
    }
    capture_merged_output_packets<0>(in_j_next[0], from_aie_words, worker_id);
    capture_merged_output_packets<1>(in_j_next[1], from_aie_words, worker_id);
    capture_merged_output_packets<2>(in_j_next[2], from_aie_words, worker_id);
    capture_merged_output_packets<3>(in_j_next[3], from_aie_words, worker_id);
    store_16lane_output_rows(next, worker_id, from_aie_words, lane_stat);
    collect_16lane_stats(lane_stat, lane_sum, lane_sum2);
}

void run_worker_strip(const float* current,
                      float* next,
                      int worker_id,
                      hls::stream<plio_word_t> out_j[srad_cfg::kLanesPerTopPl],
                      hls::stream<plio_word_t> in_j_next[srad_cfg::kMergedOutputsPerTopPl],
                      float q0sqr,
                      float& next_sum,
                      float& next_sum2) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
#pragma HLS ALLOCATION instances=run_one_strip_batch_16lanes limit=1 function
    float lane_sum[srad_cfg::kLanesPerTopPl];
    float lane_sum2[srad_cfg::kLanesPerTopPl];
#pragma HLS ARRAY_PARTITION variable=lane_sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane_sum2 complete dim=1

    run_one_strip_batch_16lanes(current, next, out_j, in_j_next,
                                worker_id, q0sqr, lane_sum, lane_sum2);

    float sum_acc = 0.0f;
    float sum2_acc = 0.0f;
    for (int lane = 0; lane < srad_cfg::kLanesPerTopPl; ++lane) {
#pragma HLS PIPELINE off
        sum_acc += lane_sum[lane];
        sum2_acc += lane_sum2[lane];
    }
    next_sum = sum_acc;
    next_sum2 = sum2_acc;
}

void copy_final_worker_region_to_output(const float* src,
                                        float* output,
                                        int worker_id) {
#pragma HLS INLINE off
    const int col_base = worker_col_base(worker_id);
    const int row_base_global = worker_row_base(worker_id);
    for (int local_row = 0;
         local_row < srad_cfg::kRowsPerRowBlock;
         ++local_row) {
        const int board_row = row_base_global + local_row;
        if (real_board_row(board_row)) {
            const int row_base = board_row * srad_cfg::kBoardCols + col_base;
            for (int elem = 0; elem < kWorkerDataElems; ++elem) {
#pragma HLS PIPELINE II=1
                output[row_base + elem] = src[row_base + elem];
            }
        }
    }
}

} // namespace

extern "C" {

void TopPL(float* image,
           float* output,
           float* debug,
           int iter_cnt,
           int worker_id,
           hls::stream<plio_word_t> out_j[srad_cfg::kLanesPerTopPl],
           hls::stream<plio_word_t> in_j_next[srad_cfg::kMergedOutputsPerTopPl],
           hls::stream<plio_word_t>& stat_to_q0,
           hls::stream<plio_word_t>& q0_from_ctrl) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 \
    max_read_burst_length=16 max_write_burst_length=16 \
    num_read_outstanding=1 num_write_outstanding=1 \
    max_widen_bitwidth=128
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 \
    max_read_burst_length=16 max_write_burst_length=16 \
    num_read_outstanding=1 num_write_outstanding=1 \
    max_widen_bitwidth=128
#pragma HLS INTERFACE m_axi port=debug offset=slave bundle=gmem2
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE axis port=stat_to_q0
#pragma HLS INTERFACE axis port=q0_from_ctrl
#pragma HLS ARRAY_PARTITION variable=out_j complete dim=1
#pragma HLS ARRAY_PARTITION variable=in_j_next complete dim=1
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=debug bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE s_axilite port=worker_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS ALLOCATION instances=run_worker_strip limit=1 function

    const int active_iters = active_iterations(iter_cnt);
    const int active_worker = clamp_worker_id(worker_id);
    write_toppl_debug(debug, active_worker, 0, 1000.0f + static_cast<float>(active_worker));
    write_toppl_debug(debug, active_worker, 1, static_cast<float>(active_iters));

    float sum = 0.0f;
    float sum2 = 0.0f;
    compute_initial_worker_stats(image, active_worker, sum, sum2);
    write_toppl_debug(debug, active_worker, 2, 1010.0f);
    stat_to_q0.write(pack_two_floats(sum, sum2));
    write_toppl_debug(debug, active_worker, 3, 1020.0f);

    for (int iter = 0; iter < srad_cfg::kBoardIterations; ++iter) {
        if (iter < active_iters) {
            const plio_word_t q0_word = q0_from_ctrl.read();
            const float q0sqr = unpack_lane0(q0_word);
            write_toppl_debug(debug, active_worker, 4, 1030.0f + static_cast<float>(iter));
            float next_sum = 0.0f;
            float next_sum2 = 0.0f;

            if ((iter & 1) == 0) {
                run_worker_strip(image,
                                 output,
                                 active_worker,
                                 out_j,
                                 in_j_next,
                                 q0sqr,
                                 next_sum,
                                 next_sum2);
            } else {
                run_worker_strip(output,
                                 image,
                                 active_worker,
                                 out_j,
                                 in_j_next,
                                 q0sqr,
                                 next_sum,
                                 next_sum2);
            }
            write_toppl_debug(debug, active_worker, 5, 1040.0f + static_cast<float>(iter));
            sum = next_sum;
            sum2 = next_sum2;
            stat_to_q0.write(pack_two_floats(sum, sum2));
            write_toppl_debug(debug, active_worker, 6, 1050.0f + static_cast<float>(iter));
        }
    }

    if ((active_iters & 1) == 0) {
        copy_final_worker_region_to_output(image, output, active_worker);
    }
    write_toppl_debug(debug, active_worker, 7, 2000.0f + static_cast<float>(active_worker));
}

}
