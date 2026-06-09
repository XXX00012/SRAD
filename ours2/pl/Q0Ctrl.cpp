#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;
constexpr int kQ0DebugBase =
    srad_cfg::kOutputElems + srad_cfg::kTopPlWorkers;

static_assert(srad_cfg::kTopPlWorkers == 4,
              "ours2 Q0Ctrl expects four TopPL workers");

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

void write_q0_debug(float* debug, int slot, float value) {
#pragma HLS INLINE
    debug[kQ0DebugBase + slot] = value;
}

void read_stats(hls::stream<plio_word_t>& stat_in_0,
                hls::stream<plio_word_t>& stat_in_1,
                hls::stream<plio_word_t>& stat_in_2,
                hls::stream<plio_word_t>& stat_in_3,
                float* debug,
                int phase,
                float& sum,
                float& sum2) {
#pragma HLS INLINE off
    const plio_word_t s0 = stat_in_0.read();
    write_q0_debug(debug, 2, 30.0f + static_cast<float>(phase));
    const plio_word_t s1 = stat_in_1.read();
    write_q0_debug(debug, 3, 40.0f + static_cast<float>(phase));
    const plio_word_t s2 = stat_in_2.read();
    write_q0_debug(debug, 4, 50.0f + static_cast<float>(phase));
    const plio_word_t s3 = stat_in_3.read();
    write_q0_debug(debug, 5, 60.0f + static_cast<float>(phase));

    sum = unpack_lane0(s0) + unpack_lane0(s1) +
          unpack_lane0(s2) + unpack_lane0(s3);
    sum2 = unpack_lane1(s0) + unpack_lane1(s1) +
           unpack_lane1(s2) + unpack_lane1(s3);
}

void broadcast_q0(float q0sqr,
                  hls::stream<plio_word_t>& q0_out_0,
                  hls::stream<plio_word_t>& q0_out_1,
                  hls::stream<plio_word_t>& q0_out_2,
                  hls::stream<plio_word_t>& q0_out_3,
                  float* debug,
                  int phase) {
#pragma HLS INLINE off
    const plio_word_t q0_word = pack_two_floats(q0sqr, 0.0f);
    q0_out_0.write(q0_word);
    write_q0_debug(debug, 2, 80.0f + static_cast<float>(phase));
    q0_out_1.write(q0_word);
    write_q0_debug(debug, 3, 90.0f + static_cast<float>(phase));
    q0_out_2.write(q0_word);
    write_q0_debug(debug, 4, 100.0f + static_cast<float>(phase));
    q0_out_3.write(q0_word);
    write_q0_debug(debug, 5, 110.0f + static_cast<float>(phase));
}

} // namespace

extern "C" {

void Q0Ctrl(float* debug,
            int iter_cnt,
            hls::stream<plio_word_t>& stat_in_0,
            hls::stream<plio_word_t>& stat_in_1,
            hls::stream<plio_word_t>& stat_in_2,
            hls::stream<plio_word_t>& stat_in_3,
            hls::stream<plio_word_t>& q0_out_0,
            hls::stream<plio_word_t>& q0_out_1,
            hls::stream<plio_word_t>& q0_out_2,
            hls::stream<plio_word_t>& q0_out_3) {
#pragma HLS INTERFACE m_axi port=debug offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=debug bundle=control
#pragma HLS INTERFACE s_axilite port=iter_cnt bundle=control
#pragma HLS INTERFACE axis port=stat_in_0
#pragma HLS INTERFACE axis port=stat_in_1
#pragma HLS INTERFACE axis port=stat_in_2
#pragma HLS INTERFACE axis port=stat_in_3
#pragma HLS INTERFACE axis port=q0_out_0
#pragma HLS INTERFACE axis port=q0_out_1
#pragma HLS INTERFACE axis port=q0_out_2
#pragma HLS INTERFACE axis port=q0_out_3
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int active_iters = active_iterations(iter_cnt);
    write_q0_debug(debug, 0, 10.0f);
    write_q0_debug(debug, 1, static_cast<float>(active_iters));

    for (int phase = 0; phase <= srad_cfg::kSradIterations; ++phase) {
        if (phase <= active_iters) {
            write_q0_debug(debug, 0, 20.0f + static_cast<float>(phase));
            write_q0_debug(debug, 1, static_cast<float>(phase));
            float sum = 0.0f;
            float sum2 = 0.0f;
            read_stats(stat_in_0,
                       stat_in_1,
                       stat_in_2,
                       stat_in_3,
                       debug,
                       phase,
                       sum,
                       sum2);

            if (phase < active_iters) {
                const float q0sqr = compute_q0sqr_from_sums(sum, sum2);
                write_q0_debug(debug, 0, 70.0f + static_cast<float>(phase));
                broadcast_q0(q0sqr,
                             q0_out_0,
                             q0_out_1,
                             q0_out_2,
                             q0_out_3,
                             debug,
                             phase);
            }
        }
    }

    write_q0_debug(debug, 0, 200.0f);
}

}
