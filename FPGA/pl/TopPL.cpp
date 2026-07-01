#include "../aie/Config.h"

#include <hls_stream.h>

namespace {

inline int clamp_int(int value, int lo, int hi) {
#pragma HLS INLINE
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

inline int image_index(int row, int col) {
#pragma HLS INLINE
    return row + srad_cfg::kRows * col;
}

inline int tile_start_row(int tile) {
#pragma HLS INLINE
    return (tile / srad_cfg::kColTiles) * srad_cfg::kTileRows;
}

inline int tile_start_col(int tile) {
#pragma HLS INLINE
    return (tile % srad_cfg::kColTiles) * srad_cfg::kTileCols;
}

void stream_q0_input(const float* image, hls::stream<float>& out_compute) {
#pragma HLS INLINE off
    for (int col = 0; col < srad_cfg::kCols; ++col) {
        for (int row = 0; row < srad_cfg::kRows; ++row) {
#pragma HLS PIPELINE II=1
            out_compute.write(image[image_index(row, col)]);
        }
    }
}

void stream_tile_input(const float* image,
                       int tile,
                       hls::stream<float>& out_compute) {
#pragma HLS INLINE off
    const int start_row = tile_start_row(tile);
    const int start_col = tile_start_col(tile);

    for (int local_row = 0;
         local_row < srad_cfg::kInputTileRows;
         ++local_row) {
        const int row =
            clamp_int(start_row + local_row - 1, 0, srad_cfg::kRows - 1);

        for (int local_col = 0;
             local_col < srad_cfg::kInputTileCols;
             ++local_col) {
#pragma HLS PIPELINE II=1
            const int col = clamp_int(start_col + local_col -
                                          srad_cfg::kTileLeftHaloCols,
                                      0,
                                      srad_cfg::kCols - 1);
            out_compute.write(image[image_index(row, col)]);
        }
    }
}

} // namespace

extern "C" {

void LoadFpgaV5(const float* image,
                float lambda,
                hls::stream<float>& out_params,
                hls::stream<float>& out_compute) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE axis port=out_params
#pragma HLS INTERFACE axis port=out_compute
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=lambda bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    out_params.write(lambda);
    out_params.write(0.0f);

    stream_q0_input(image, out_compute);

    for (int tile = 0; tile < srad_cfg::kTilesPerIteration; ++tile) {
        stream_tile_input(image, tile, out_compute);
    }
}

void StoreFpgaV5(hls::stream<float>& in_index,
                 hls::stream<float>& in_value,
                 float* output) {
#pragma HLS INTERFACE axis port=in_index
#pragma HLS INTERFACE axis port=in_value
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < srad_cfg::kOutputStreamElems; ++i) {
#pragma HLS PIPELINE II=1
        const float index_f = in_index.read();
        const float value = in_value.read();
        const int index = static_cast<int>(index_f);
        if (index >= 0 && index < srad_cfg::kOutputElems) {
            output[index] = value;
        }
    }
}

}
