#include "TopGraph.h"

GraphOursPLQ0 graphOursPLQ0("ours_plq0");

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

void dump_plio64_float_file(const std::string& path, const float* buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr,
                     "[aiesim] cannot open %s for write\n",
                     path.c_str());
        return;
    }

    fout << std::setprecision(9);
    for (int i = 0; i < srad_cfg::kAieOutputElems; i += 2) {
        fout << buf[i] << ' ' << buf[i + 1] << '\n';
    }
}

} // namespace

int main() {
    std::printf("[aiesim] GraphOursPLQ0: %d graph firing(s), %d output floats per firing, two-kernel srad_local_q -> srad_coeff_update\n",
                srad_cfg::kGraphRunIterations,
                srad_cfg::kOutputSampleElems);
    std::fflush(stdout);
    graphOursPLQ0.init();
    graphOursPLQ0.run(srad_cfg::kGraphRunIterations);
    graphOursPLQ0.wait();
    graphOursPLQ0.end();

    std::vector<float> raw_out_buf(srad_cfg::kAieOutputElems);
    if (!load_float_file(srad_plio_files::kJNextOut, raw_out_buf)) {
        return EXIT_FAILURE;
    }
    dump_plio64_float_file("./data/aiesim_j_next.txt", raw_out_buf.data());
    std::printf("[aiesim] wrote ./data/aiesim_j_next.txt\n");

    return EXIT_SUCCESS;
}
#endif
