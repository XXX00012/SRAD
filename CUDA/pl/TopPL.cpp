#include "../aie/Config.h"

#include <hls_stream.h>

namespace {

int clamp_index(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

int valid_tile_rows(int row0) {
    const int remain = srad_cfg::kRows - row0;
    return (remain < srad_cfg::kBlockRows) ? remain : srad_cfg::kBlockRows;
}

int valid_tile_cols(int col0) {
    const int remain = srad_cfg::kCols - col0;
    return (remain < srad_cfg::kBlockCols) ? remain : srad_cfg::kBlockCols;
}

int opencl_index(int row, int col) {
    return row + srad_cfg::kRows * col;
}

void stream_plain_tile(const float* image,
                       hls::stream<float>& out,
                       int row0,
                       int col0,
                       bool zero_outside) {
    const int rows = valid_tile_rows(row0);
    const int cols = valid_tile_cols(col0);

    for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
        for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
#pragma HLS PIPELINE II=1
            float v = 0.0f;
            if (!zero_outside || (r < rows && c < cols)) {
                const int gr = clamp_index(row0 + r, 0, srad_cfg::kRows - 1);
                const int gc = clamp_index(col0 + c, 0, srad_cfg::kCols - 1);
                v = image[opencl_index(gr, gc)];
            }
            out.write(v);
        }
    }
}

void stream_full_image(const float* image, hls::stream<float>& out) {
    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        out.write(image[i]);
    }
}

void store_plain_tile(hls::stream<float>& in,
                      float* image,
                      int row0,
                      int col0) {
    const int rows = valid_tile_rows(row0);
    const int cols = valid_tile_cols(col0);

    for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
        for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
#pragma HLS PIPELINE II=1
            const float v = in.read();
            if (r < rows && c < cols) {
                image[opencl_index(row0 + r, col0 + c)] = v;
            }
        }
    }
}

void stream_q0_packet(const float* q0_packet, hls::stream<float>& out) {
    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        out.write(q0_packet[i]);
    }
}

} // namespace

extern "C" {

void LoadPrepare(const float* image,
                 hls::stream<float>& out_j) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    stream_full_image(image, out_j);
}

void StorePrepare(hls::stream<float>& in_sums,
                  hls::stream<float>& in_sums2,
                  float* sums,
                  float* sums2) {
#pragma HLS INTERFACE axis port=in_sums
#pragma HLS INTERFACE axis port=in_sums2
#pragma HLS INTERFACE m_axi port=sums offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=sums2 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=sums bundle=control
#pragma HLS INTERFACE s_axilite port=sums2 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        sums[i] = in_sums.read();
        sums2[i] = in_sums2.read();
    }
}

void LoadReduce(const float* sums,
                const float* sums2,
                hls::stream<float>& out_sums,
                hls::stream<float>& out_sums2) {
#pragma HLS INTERFACE m_axi port=sums offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=sums2 offset=slave bundle=gmem1
#pragma HLS INTERFACE axis port=out_sums
#pragma HLS INTERFACE axis port=out_sums2
#pragma HLS INTERFACE s_axilite port=sums bundle=control
#pragma HLS INTERFACE s_axilite port=sums2 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        out_sums.write(sums[i]);
        out_sums2.write(sums2[i]);
    }
}

void StoreReduce(hls::stream<float>& in_stats,
                 float* stats) {
#pragma HLS INTERFACE axis port=in_stats
#pragma HLS INTERFACE m_axi port=stats offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=stats bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        stats[i] = in_stats.read();
    }
}

void LoadCoeff(const float* image,
               float q0sqr,
               hls::stream<float>& out_j,
               hls::stream<float>& out_q0sqr) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=out_q0sqr
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=q0sqr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float q0_packet[srad_cfg::kScalarPacketElems];
#pragma HLS ARRAY_PARTITION variable=q0_packet complete dim=1
    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        q0_packet[i] = (i == 0) ? q0sqr : 0.0f;
    }

    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            stream_q0_packet(q0_packet, out_q0sqr);
            stream_full_image(image, out_j);
        }
    }
}

void StoreCoeff(hls::stream<float>& in_c,
                hls::stream<float>& in_dN,
                hls::stream<float>& in_dS,
                hls::stream<float>& in_dW,
                hls::stream<float>& in_dE,
                float* c_plane,
                float* dN_plane,
                float* dS_plane,
                float* dW_plane,
                float* dE_plane) {
#pragma HLS INTERFACE axis port=in_c
#pragma HLS INTERFACE axis port=in_dN
#pragma HLS INTERFACE axis port=in_dS
#pragma HLS INTERFACE axis port=in_dW
#pragma HLS INTERFACE axis port=in_dE
#pragma HLS INTERFACE m_axi port=c_plane offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=dN_plane offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dS_plane offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dW_plane offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=dE_plane offset=slave bundle=gmem4
#pragma HLS INTERFACE s_axilite port=c_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dN_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dS_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dW_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dE_plane bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            const int row0 = tr * srad_cfg::kBlockRows;
            const int col0 = tc * srad_cfg::kBlockCols;
            const int rows = valid_tile_rows(row0);
            const int cols = valid_tile_cols(col0);
            for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
                for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
#pragma HLS PIPELINE II=1
                    const float cv = in_c.read();
                    const float dNv = in_dN.read();
                    const float dSv = in_dS.read();
                    const float dWv = in_dW.read();
                    const float dEv = in_dE.read();
                    if (r < rows && c < cols) {
                        const int idx = opencl_index(row0 + r, col0 + c);
                        c_plane[idx] = cv;
                        dN_plane[idx] = dNv;
                        dS_plane[idx] = dSv;
                        dW_plane[idx] = dWv;
                        dE_plane[idx] = dEv;
                    }
                }
            }
        }
    }
}

void LoadUpdate(const float* image,
                const float* c_plane,
                const float* dN_plane,
                const float* dS_plane,
                const float* dW_plane,
                const float* dE_plane,
                float lambda,
                hls::stream<float>& out_j,
                hls::stream<float>& out_c,
                hls::stream<float>& out_dN,
                hls::stream<float>& out_dS,
                hls::stream<float>& out_dW,
                hls::stream<float>& out_dE,
                hls::stream<float>& out_lambda) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=c_plane offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=dN_plane offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=dS_plane offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=dW_plane offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=dE_plane offset=slave bundle=gmem5
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=out_c
#pragma HLS INTERFACE axis port=out_dN
#pragma HLS INTERFACE axis port=out_dS
#pragma HLS INTERFACE axis port=out_dW
#pragma HLS INTERFACE axis port=out_dE
#pragma HLS INTERFACE axis port=out_lambda
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=c_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dN_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dS_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dW_plane bundle=control
#pragma HLS INTERFACE s_axilite port=dE_plane bundle=control
#pragma HLS INTERFACE s_axilite port=lambda bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            const int row0 = tr * srad_cfg::kBlockRows;
            const int col0 = tc * srad_cfg::kBlockCols;
            for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
                out_lambda.write((i == 0) ? lambda : 0.0f);
            }
            stream_plain_tile(image, out_j, row0, col0, false);
            stream_full_image(c_plane, out_c);
            stream_plain_tile(dN_plane, out_dN, row0, col0, false);
            stream_plain_tile(dS_plane, out_dS, row0, col0, false);
            stream_plain_tile(dW_plane, out_dW, row0, col0, false);
            stream_plain_tile(dE_plane, out_dE, row0, col0, false);
        }
    }
}

void StoreUpdate(hls::stream<float>& in_j_next,
                 float* output) {
#pragma HLS INTERFACE axis port=in_j_next
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            const int row0 = tr * srad_cfg::kBlockRows;
            const int col0 = tc * srad_cfg::kBlockCols;
            store_plain_tile(in_j_next, output, row0, col0);
        }
    }
}

}
