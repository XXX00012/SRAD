#include "../aie/Config.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

namespace {

using plio_word_t = ap_uint<64>;

static_assert(srad_cfg::kCudaBlockElems % 2 == 0,
              "64-bit PLIO packing requires an even block element count");
static_assert(srad_cfg::kMetaPacketElems % 2 == 0,
              "64-bit PLIO packing requires an even metadata packet length");

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

int active_blocks(int block_count) {
#pragma HLS INLINE
    int blocks = block_count;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > srad_cfg::kCudaBlocks) {
        blocks = srad_cfg::kCudaBlocks;
    }
    return blocks;
}

int valid_count_for_block(int block) {
#pragma HLS INLINE
    const int base = block * srad_cfg::kCudaBlockElems;
    int remaining = srad_cfg::kBoardPixels - base;
    if (remaining < 0) {
        remaining = 0;
    }
    if (remaining > srad_cfg::kCudaBlockElems) {
        remaining = srad_cfg::kCudaBlockElems;
    }
    return remaining;
}

float image_value_or_zero(const float* image, int linear_index) {
#pragma HLS INLINE
    if (linear_index < 0 || linear_index >= srad_cfg::kBoardPixels) {
        return 0.0f;
    }
    return image[linear_index];
}

float neighbor_value_or_zero(const float* image, int linear_index, int row_delta, int col_delta) {
#pragma HLS INLINE
    const int row = (linear_index / srad_cfg::kBoardCols) + row_delta;
    const int col = (linear_index % srad_cfg::kBoardCols) + col_delta;
    if (row < 0 || row >= srad_cfg::kBoardRows ||
        col < 0 || col >= srad_cfg::kBoardCols) {
        return 0.0f;
    }
    return image[row * srad_cfg::kBoardCols + col];
}

float coeff_neighbor_or_zero(const float* coeff, int linear_index, int row_delta, int col_delta) {
#pragma HLS INLINE
    const int row = (linear_index / srad_cfg::kBoardCols) + row_delta;
    const int col = (linear_index % srad_cfg::kBoardCols) + col_delta;
    if (row < 0 || row >= srad_cfg::kBoardRows ||
        col < 0 || col >= srad_cfg::kBoardCols) {
        return 0.0f;
    }
    return coeff[row * srad_cfg::kBoardCols + col];
}

void write_meta_packet(float first_value, hls::stream<plio_word_t>& out) {
#pragma HLS INLINE off
    out.write(pack_two_floats(first_value, 0.0f));
    for (int word = 1; word < (srad_cfg::kMetaPacketElems / 2); ++word) {
#pragma HLS PIPELINE II=1
        out.write(pack_two_floats(0.0f, 0.0f));
    }
}

} // namespace

extern "C" {

void PreparePL(const float* image,
               float* sums,
               float* sums2,
               int block_count,
               hls::stream<plio_word_t>& out_i,
               hls::stream<plio_word_t>& in_sums,
               hls::stream<plio_word_t>& in_sums2) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=sums offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=sums2 offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=sums bundle=control
#pragma HLS INTERFACE s_axilite port=sums2 bundle=control
#pragma HLS INTERFACE s_axilite port=block_count bundle=control
#pragma HLS INTERFACE axis port=out_i
#pragma HLS INTERFACE axis port=in_sums
#pragma HLS INTERFACE axis port=in_sums2
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int blocks = active_blocks(block_count);
    for (int block = 0; block < blocks; ++block) {
        const int base = block * srad_cfg::kCudaBlockElems;

        for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
            const int idx0 = base + (2 * word);
            const int idx1 = idx0 + 1;
            out_i.write(pack_two_floats(
                image_value_or_zero(image, idx0),
                image_value_or_zero(image, idx1)));
        }

        for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
            const plio_word_t sum_word = in_sums.read();
            const plio_word_t sum2_word = in_sums2.read();
            const int idx0 = base + (2 * word);
            const int idx1 = idx0 + 1;
            if (idx0 < srad_cfg::kBoardPixels) {
                sums[idx0] = unpack_lane0(sum_word);
                sums2[idx0] = unpack_lane0(sum2_word);
            }
            if (idx1 < srad_cfg::kBoardPixels) {
                sums[idx1] = unpack_lane1(sum_word);
                sums2[idx1] = unpack_lane1(sum2_word);
            }
        }
    }
}

void ReducePL(const float* sums,
              const float* sums2,
              float* partials,
              int block_count,
              hls::stream<plio_word_t>& out_packet,
              hls::stream<plio_word_t>& in_partial) {
#pragma HLS INTERFACE m_axi port=sums offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=sums2 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=partials offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=sums bundle=control
#pragma HLS INTERFACE s_axilite port=sums2 bundle=control
#pragma HLS INTERFACE s_axilite port=partials bundle=control
#pragma HLS INTERFACE s_axilite port=block_count bundle=control
#pragma HLS INTERFACE axis port=out_packet
#pragma HLS INTERFACE axis port=in_partial
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int blocks = active_blocks(block_count);
    for (int block = 0; block < blocks; ++block) {
        const int base = block * srad_cfg::kCudaBlockElems;
        write_meta_packet(static_cast<float>(valid_count_for_block(block)), out_packet);

        for (int tx = 0; tx < srad_cfg::kCudaBlockElems; ++tx) {
#pragma HLS PIPELINE II=1
            const int idx = base + tx;
            const float sum_value =
                (idx < srad_cfg::kBoardPixels) ? sums[idx] : 0.0f;
            const float sum2_value =
                (idx < srad_cfg::kBoardPixels) ? sums2[idx] : 0.0f;
            out_packet.write(pack_two_floats(sum_value, sum2_value));
        }

        const plio_word_t partial_word = in_partial.read();
        partials[(2 * block) + 0] = unpack_lane0(partial_word);
        partials[(2 * block) + 1] = unpack_lane1(partial_word);
    }
}

void CoeffPL(const float* image,
             float* dN,
             float* dS,
             float* dW,
             float* dE,
             float* coeff,
             float q0sqr,
             int block_count,
             hls::stream<plio_word_t>& out_neighbors,
             hls::stream<plio_word_t>& out_q0,
             hls::stream<plio_word_t>& in_dN,
             hls::stream<plio_word_t>& in_dS,
             hls::stream<plio_word_t>& in_dW,
             hls::stream<plio_word_t>& in_dE,
             hls::stream<plio_word_t>& in_c) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=dN offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dS offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dW offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=dE offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=coeff offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=dN bundle=control
#pragma HLS INTERFACE s_axilite port=dS bundle=control
#pragma HLS INTERFACE s_axilite port=dW bundle=control
#pragma HLS INTERFACE s_axilite port=dE bundle=control
#pragma HLS INTERFACE s_axilite port=coeff bundle=control
#pragma HLS INTERFACE s_axilite port=q0sqr bundle=control
#pragma HLS INTERFACE s_axilite port=block_count bundle=control
#pragma HLS INTERFACE axis port=out_neighbors
#pragma HLS INTERFACE axis port=out_q0
#pragma HLS INTERFACE axis port=in_dN
#pragma HLS INTERFACE axis port=in_dS
#pragma HLS INTERFACE axis port=in_dW
#pragma HLS INTERFACE axis port=in_dE
#pragma HLS INTERFACE axis port=in_c
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int blocks = active_blocks(block_count);
    for (int block = 0; block < blocks; ++block) {
        const int base = block * srad_cfg::kCudaBlockElems;

        for (int plane = 0; plane < srad_cfg::kCoeffInputPlanes; ++plane) {
            for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
                const int idx0 = base + (2 * word);
                const int idx1 = idx0 + 1;
                float lane0 = 0.0f;
                float lane1 = 0.0f;
                if (plane == 0) {
                    lane0 = image_value_or_zero(image, idx0);
                    lane1 = image_value_or_zero(image, idx1);
                } else if (plane == 1) {
                    lane0 = neighbor_value_or_zero(image, idx0, -1, 0);
                    lane1 = neighbor_value_or_zero(image, idx1, -1, 0);
                } else if (plane == 2) {
                    lane0 = neighbor_value_or_zero(image, idx0, 1, 0);
                    lane1 = neighbor_value_or_zero(image, idx1, 1, 0);
                } else if (plane == 3) {
                    lane0 = neighbor_value_or_zero(image, idx0, 0, -1);
                    lane1 = neighbor_value_or_zero(image, idx1, 0, -1);
                } else {
                    lane0 = neighbor_value_or_zero(image, idx0, 0, 1);
                    lane1 = neighbor_value_or_zero(image, idx1, 0, 1);
                }
                out_neighbors.write(pack_two_floats(lane0, lane1));
            }
        }
        write_meta_packet(q0sqr, out_q0);

        for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
            const plio_word_t dN_word = in_dN.read();
            const plio_word_t dS_word = in_dS.read();
            const plio_word_t dW_word = in_dW.read();
            const plio_word_t dE_word = in_dE.read();
            const plio_word_t c_word = in_c.read();
            const int idx0 = base + (2 * word);
            const int idx1 = idx0 + 1;
            if (idx0 < srad_cfg::kBoardPixels) {
                dN[idx0] = unpack_lane0(dN_word);
                dS[idx0] = unpack_lane0(dS_word);
                dW[idx0] = unpack_lane0(dW_word);
                dE[idx0] = unpack_lane0(dE_word);
                coeff[idx0] = unpack_lane0(c_word);
            }
            if (idx1 < srad_cfg::kBoardPixels) {
                dN[idx1] = unpack_lane1(dN_word);
                dS[idx1] = unpack_lane1(dS_word);
                dW[idx1] = unpack_lane1(dW_word);
                dE[idx1] = unpack_lane1(dE_word);
                coeff[idx1] = unpack_lane1(c_word);
            }
        }
    }
}

void UpdatePL(const float* image,
              const float* dN,
              const float* dS,
              const float* dW,
              const float* dE,
              const float* coeff,
              float* output,
              float lambda_value,
              int block_count,
              hls::stream<plio_word_t>& out_update,
              hls::stream<plio_word_t>& out_meta,
              hls::stream<plio_word_t>& in_i_next) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=dN offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dS offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dW offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=dE offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=coeff offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem6
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=dN bundle=control
#pragma HLS INTERFACE s_axilite port=dS bundle=control
#pragma HLS INTERFACE s_axilite port=dW bundle=control
#pragma HLS INTERFACE s_axilite port=dE bundle=control
#pragma HLS INTERFACE s_axilite port=coeff bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=lambda_value bundle=control
#pragma HLS INTERFACE s_axilite port=block_count bundle=control
#pragma HLS INTERFACE axis port=out_update
#pragma HLS INTERFACE axis port=out_meta
#pragma HLS INTERFACE axis port=in_i_next
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int blocks = active_blocks(block_count);
    for (int block = 0; block < blocks; ++block) {
        const int base = block * srad_cfg::kCudaBlockElems;

        for (int plane = 0; plane < srad_cfg::kUpdateInputPlanes; ++plane) {
            for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
                const int idx0 = base + (2 * word);
                const int idx1 = idx0 + 1;
                float lane0 = 0.0f;
                float lane1 = 0.0f;
                if (plane == 0) {
                    lane0 = image_value_or_zero(image, idx0);
                    lane1 = image_value_or_zero(image, idx1);
                } else if (plane == 1) {
                    lane0 = (idx0 < srad_cfg::kBoardPixels) ? dN[idx0] : 0.0f;
                    lane1 = (idx1 < srad_cfg::kBoardPixels) ? dN[idx1] : 0.0f;
                } else if (plane == 2) {
                    lane0 = (idx0 < srad_cfg::kBoardPixels) ? dS[idx0] : 0.0f;
                    lane1 = (idx1 < srad_cfg::kBoardPixels) ? dS[idx1] : 0.0f;
                } else if (plane == 3) {
                    lane0 = (idx0 < srad_cfg::kBoardPixels) ? dW[idx0] : 0.0f;
                    lane1 = (idx1 < srad_cfg::kBoardPixels) ? dW[idx1] : 0.0f;
                } else if (plane == 4) {
                    lane0 = (idx0 < srad_cfg::kBoardPixels) ? dE[idx0] : 0.0f;
                    lane1 = (idx1 < srad_cfg::kBoardPixels) ? dE[idx1] : 0.0f;
                } else if (plane == 5) {
                    lane0 = (idx0 < srad_cfg::kBoardPixels) ? coeff[idx0] : 0.0f;
                    lane1 = (idx1 < srad_cfg::kBoardPixels) ? coeff[idx1] : 0.0f;
                } else if (plane == 6) {
                    lane0 = coeff_neighbor_or_zero(coeff, idx0, 1, 0);
                    lane1 = coeff_neighbor_or_zero(coeff, idx1, 1, 0);
                } else {
                    lane0 = coeff_neighbor_or_zero(coeff, idx0, 0, 1);
                    lane1 = coeff_neighbor_or_zero(coeff, idx1, 0, 1);
                }
                out_update.write(pack_two_floats(lane0, lane1));
            }
        }
        write_meta_packet(lambda_value, out_meta);

        for (int word = 0; word < srad_cfg::kCudaBlockElems / 2; ++word) {
#pragma HLS PIPELINE II=1
            const plio_word_t next_word = in_i_next.read();
            const int idx0 = base + (2 * word);
            const int idx1 = idx0 + 1;
            if (idx0 < srad_cfg::kBoardPixels) {
                output[idx0] = unpack_lane0(next_word);
            }
            if (idx1 < srad_cfg::kBoardPixels) {
                output[idx1] = unpack_lane1(next_word);
            }
        }
    }
}

}
