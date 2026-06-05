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

int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

int last_opencl_block_count(int no, int grid_dim) {
    return srad_cfg::kOpenClReductionThreads -
           (grid_dim * srad_cfg::kOpenClReductionThreads - no);
}

int largest_opencl_reduction_pow2(int n) {
    int df = 1;
    for (int i = 2; i <= srad_cfg::kOpenClReductionThreads; i *= 2) {
        if (n >= i) {
            df = i;
        }
    }
    return df;
}

void reduce_full_opencl_block(float* psum, float* psum2) {
    for (int i = 2; i <= srad_cfg::kOpenClReductionThreads; i *= 2) {
        for (int tx = i - 1; tx < srad_cfg::kOpenClReductionThreads; tx += i) {
#pragma HLS PIPELINE II=1
            psum[tx] = psum[tx] + psum[tx - i / 2];
            psum2[tx] = psum2[tx] + psum2[tx - i / 2];
        }
    }
}

void opencl_v0_reduce_stats(const float* image, float& sum, float& sum2) {
    constexpr int kThreads = srad_cfg::kOpenClReductionThreads;
    float sums[srad_cfg::kPixels];
    float sums2[srad_cfg::kPixels];

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        const float v = image[i];
        sums[i] = v;
        sums2[i] = v * v;
    }

    int no = srad_cfg::kPixels;
    int mul = 1;
    int grid_dim = ceil_div(no, kThreads);

    while (grid_dim != 0) {
        const int nf = last_opencl_block_count(no, grid_dim);

        for (int bx = 0; bx < grid_dim; ++bx) {
            float psum[kThreads];
            float psum2[kThreads];

            for (int tx = 0; tx < kThreads; ++tx) {
#pragma HLS PIPELINE II=1
                const int ei = bx * kThreads + tx;
                if (ei < no) {
                    const int src = ei * mul;
                    psum[tx] = sums[src];
                    psum2[tx] = sums2[src];
                }
            }

            const int dst = bx * mul * kThreads;
            if (nf == kThreads || bx != grid_dim - 1) {
                reduce_full_opencl_block(psum, psum2);
                sums[dst] = psum[kThreads - 1];
                sums2[dst] = psum2[kThreads - 1];
            } else {
                const int df = largest_opencl_reduction_pow2(nf);
                for (int i = 2; i <= df; i *= 2) {
                    for (int tx = i - 1; tx < df; tx += i) {
#pragma HLS PIPELINE II=1
                        psum[tx] = psum[tx] + psum[tx - i / 2];
                        psum2[tx] = psum2[tx] + psum2[tx - i / 2];
                    }
                }

                const int tx = df - 1;
                for (int i = bx * kThreads + df; i < bx * kThreads + nf; ++i) {
#pragma HLS PIPELINE II=1
                    psum[tx] = psum[tx] + sums[i];
                    psum2[tx] = psum2[tx] + sums2[i];
                }

                sums[dst] = psum[tx];
                sums2[dst] = psum2[tx];
            }
        }

        no = grid_dim;
        if (grid_dim == 1) {
            grid_dim = 0;
        } else {
            mul *= kThreads;
            grid_dim = ceil_div(no, kThreads);
        }
    }

    sum = sums[0];
    sum2 = sums2[0];
}

float compute_q0sqr(const float* image) {
    float sum = 0.0f;
    float sum2 = 0.0f;
    opencl_v0_reduce_stats(image, sum, sum2);

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    const float mean2 = mean * mean;
    return variance / mean2;
}

void stream_full_image(const float* image, hls::stream<float>& out) {
    for (int i = 0; i < srad_cfg::kPixels; ++i) {
#pragma HLS PIPELINE II=1
        out.write(image[i]);
    }
}

void stream_plain_tile(const float* image,
                       hls::stream<float>& out,
                       int row0,
                       int col0) {
    for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
        for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
#pragma HLS PIPELINE II=1
            const int gr = clamp_index(row0 + r, 0, srad_cfg::kRows - 1);
            const int gc = clamp_index(col0 + c, 0, srad_cfg::kCols - 1);
            out.write(image[opencl_index(gr, gc)]);
        }
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

void stream_q0_packet(float q0sqr, hls::stream<float>& out) {
    for (int i = 0; i < srad_cfg::kScalarPacketElems; ++i) {
#pragma HLS PIPELINE II=1
        out.write((i == 0) ? q0sqr : 0.0f);
    }
}

} // namespace

extern "C" {

void TopPL(const float* image,
           hls::stream<float>& out_j,
           hls::stream<float>& out_q0sqr) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_j
#pragma HLS INTERFACE axis port=out_q0sqr
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const float q0sqr = compute_q0sqr(image);
    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            stream_q0_packet(q0sqr, out_q0sqr);
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
            stream_plain_tile(image, out_j, row0, col0);
            stream_full_image(c_plane, out_c);
            stream_plain_tile(dN_plane, out_dN, row0, col0);
            stream_plain_tile(dS_plane, out_dS, row0, col0);
            stream_plain_tile(dW_plane, out_dW, row0, col0);
            stream_plain_tile(dE_plane, out_dE, row0, col0);
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
