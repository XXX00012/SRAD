#include "TopGraph.h"

#include <cstddef>

GraphOursPLQ0 graphOursPLQ0("ours_undivide_plq0");

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

void pack_blocks(const std::vector<float>& image, float* blocks) {
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

void unpack_blocks(const float* blocks, std::vector<float>& image) {
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

bool write_stream_file(const std::string& path, const float* buf, int count) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr,
                     "[aiesim] cannot open %s for write\n",
                     path.c_str());
        return false;
    }

    for (int i = 0; i < count; ++i) {
        fout << buf[i] << '\n';
    }
    return true;
}

bool write_scalar_packet_file(const std::string& path, float value) {
    std::vector<float> packets(srad_cfg::kBlockCount *
                               srad_cfg::kScalarPacketElems,
                               0.0f);
    for (int blk = 0; blk < srad_cfg::kBlockCount; ++blk) {
        packets[blk * srad_cfg::kScalarPacketElems] = value;
    }
    return write_stream_file(path, packets.data(),
                             static_cast<int>(packets.size()));
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

    std::vector<float> j_buf(srad_cfg::kPixels);
    pack_blocks(image, j_buf.data());
    if (!write_stream_file(srad_plio_files::kJIn,
                           j_buf.data(),
                           srad_cfg::kPixels) ||
        !write_stream_file(srad_plio_files::kJUpdateIn,
                           j_buf.data(),
                           srad_cfg::kPixels) ||
        !write_scalar_packet_file(srad_plio_files::kQ0In, q0sqr) ||
        !write_scalar_packet_file(srad_plio_files::kLambdaIn, lambda)) {
        return EXIT_FAILURE;
    }

    std::printf("[aiesim] GraphOursUndividePLQ0: %d graph firing(s), %dx%d floats per firing, fused full-tile pressure block on 16x16 data\n",
                srad_cfg::kBlockCount,
                srad_cfg::kBlockRows,
                srad_cfg::kBlockCols);
    std::printf("[aiesim] stack experiment: 16x16 data, 16x16 tile, single-row direct fused kernel, no manual spill; one graph firing moves and updates 256 floats while SIMD math uses v8float lanes\n");
    std::printf("[aiesim] pressure group: %d rows, %d v8float vectors per inner group\n",
                srad_cfg::kPressureRowsPerGroup,
                srad_cfg::kPressureVectorsPerGroup);
    std::printf("[aiesim] update: standard J_next = J + 0.25 * lambda * D\n");
    std::fflush(stdout);
    graphOursPLQ0.init();
    graphOursPLQ0.run(srad_cfg::kBlockCount);
    graphOursPLQ0.wait();
    graphOursPLQ0.end();

    std::vector<float> out_buf(srad_cfg::kPixels);
    if (!load_float_file(srad_plio_files::kJNextOut, out_buf)) {
        return EXIT_FAILURE;
    }
    std::vector<float> out_image(srad_cfg::kPixels);
    unpack_blocks(out_buf.data(), out_image);
    dump_float_file("./data/aiesim_j_next.txt", out_image.data());
    std::printf("[aiesim] wrote ./data/aiesim_j_next.txt\n");

    return EXIT_SUCCESS;
}
#endif
