#include "TopGraph.h"

#include <cstddef>

GraphOursPLQ0 graphOursPLQ0("ours_compute_plq0");

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool load_float_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[aiesim] cannot open %s\n", path.c_str());
        return false;
    }

    float v = 0.0f;
    int cnt = 0;
    while (fin >> v) {
        if (cnt >= static_cast<int>(buf.size())) break;
        buf[cnt++] = v;
    }

    if (cnt != static_cast<int>(buf.size())) {
        std::fprintf(stderr,
                     "[aiesim] %s element count mismatch: got %d, expect %zu\n",
                     path.c_str(), cnt, buf.size());
        return false;
    }
    return true;
}

bool load_scalar_file(const std::string& path, float& value) {
    std::ifstream fin(path);
    float v = 0.0f;
    if (!(fin >> v)) {
        std::fprintf(stderr,
                     "[aiesim] cannot read scalar from %s\n",
                     path.c_str());
        return false;
    }
    value = v;
    return true;
}

void dump_float_file(const std::string& path, const float* buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr,
                     "[aiesim] cannot open %s for write\n",
                     path.c_str());
        return;
    }

    for (int r = 0; r < srad_cfg::kRows; ++r) {
        for (int c = 0; c < srad_cfg::kCols; ++c) {
            if (c) fout << ' ';
            fout << buf[r * srad_cfg::kCols + c];
        }
        fout << '\n';
    }
}

void pack_8x8_blocks(const std::vector<float>& image, float* blocks) {
    int out = 0;
    for (int br = 0; br < srad_cfg::kRows; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < srad_cfg::kCols; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                const int src = (br + r) * srad_cfg::kCols + bc;
                std::memcpy(blocks + out,
                            image.data() + src,
                            srad_cfg::kBlockCols * sizeof(float));
                out += srad_cfg::kBlockCols;
            }
        }
    }
}

void unpack_8x8_blocks(const float* blocks, std::vector<float>& image) {
    int in = 0;
    for (int br = 0; br < srad_cfg::kRows; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < srad_cfg::kCols; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                const int dst = (br + r) * srad_cfg::kCols + bc;
                std::memcpy(image.data() + dst,
                            blocks + in,
                            srad_cfg::kBlockCols * sizeof(float));
                in += srad_cfg::kBlockCols;
            }
        }
    }
}

float* gmio_malloc_or_null(std::size_t bytes) {
    return reinterpret_cast<float*>(adf::GMIO::malloc(bytes));
}

} // namespace

int main() {
    std::vector<float> image(srad_cfg::kPixels);
    if (!load_float_file("./data/input_32x32.txt", image)) {
        return EXIT_FAILURE;
    }

    float lambda = srad_cfg::kLambdaDefault;
    if (!load_scalar_file("./data/lambda.txt", lambda)) {
        return EXIT_FAILURE;
    }

    float q0sqr = 0.0f;
    if (!load_scalar_file("./data/q0sqr.txt", q0sqr)) {
        return EXIT_FAILURE;
    }

    auto* j_buf = gmio_malloc_or_null(srad_cfg::kImageBytes);
    auto* q0_buf = gmio_malloc_or_null(srad_cfg::kScalarPacketsPerImageBytes);
    auto* lambda_buf =
        gmio_malloc_or_null(srad_cfg::kScalarPacketsPerImageBytes);
    auto* out_buf = gmio_malloc_or_null(srad_cfg::kImageBytes);

    if (!j_buf || !q0_buf || !lambda_buf || !out_buf) {
        std::fprintf(stderr, "[aiesim] GMIO::malloc failed\n");
        if (j_buf) adf::GMIO::free(j_buf);
        if (q0_buf) adf::GMIO::free(q0_buf);
        if (lambda_buf) adf::GMIO::free(lambda_buf);
        if (out_buf) adf::GMIO::free(out_buf);
        return EXIT_FAILURE;
    }

    pack_8x8_blocks(image, j_buf);
    std::memset(q0_buf, 0, srad_cfg::kScalarPacketsPerImageBytes);
    std::memset(lambda_buf, 0, srad_cfg::kScalarPacketsPerImageBytes);
    for (int blk = 0; blk < srad_cfg::kBlockCount; ++blk) {
        q0_buf[blk * srad_cfg::kScalarPacketElems] = q0sqr;
        lambda_buf[blk * srad_cfg::kScalarPacketElems] = lambda;
    }
    std::memset(out_buf, 0, srad_cfg::kImageBytes);

    std::printf("[aiesim] GraphOursComputePLQ0: %d graph firing(s), %dx%d floats per firing, local differences -> original five-division coeff/update\n",
                srad_cfg::kBlockCount,
                srad_cfg::kBlockRows,
                srad_cfg::kBlockCols);
    std::fflush(stdout);
    graphOursPLQ0.init();
    graphOursPLQ0.out_j_next.aie2gm_nb(out_buf, srad_cfg::kImageBytes);
    graphOursPLQ0.in_j.gm2aie_nb(j_buf, srad_cfg::kImageBytes);
    graphOursPLQ0.in_q0sqr.gm2aie_nb(q0_buf,
                                      srad_cfg::kScalarPacketsPerImageBytes);
    graphOursPLQ0.in_lambda.gm2aie_nb(lambda_buf,
                                      srad_cfg::kScalarPacketsPerImageBytes);
    graphOursPLQ0.in_j.wait();
    graphOursPLQ0.in_q0sqr.wait();
    graphOursPLQ0.in_lambda.wait();
    graphOursPLQ0.run(srad_cfg::kBlockCount);
    graphOursPLQ0.wait();
    graphOursPLQ0.out_j_next.wait();
    graphOursPLQ0.end();

    std::vector<float> out_image(srad_cfg::kPixels);
    unpack_8x8_blocks(out_buf, out_image);
    dump_float_file("./data/aiesim_j_next.txt", out_image.data());
    std::printf("[aiesim] wrote ./data/aiesim_j_next.txt\n");

    adf::GMIO::free(j_buf);
    adf::GMIO::free(q0_buf);
    adf::GMIO::free(lambda_buf);
    adf::GMIO::free(out_buf);
    return EXIT_SUCCESS;
}
#endif
