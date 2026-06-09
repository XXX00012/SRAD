#include "TopGraph.h"

GraphOursPLQ0 graphOursPLQ0("ours_plq0");

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>
#include <cstdlib>
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

void dump_float_file(const std::string& path, const float* buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr,
                     "[aiesim] cannot open %s for write\n",
                     path.c_str());
        return;
    }

    for (int r = 0; r < srad_cfg::kOutputRows; ++r) {
        for (int c = 0; c < srad_cfg::kOutputCols; ++c) {
            if (c) fout << ' ';
            fout << buf[r * srad_cfg::kOutputCols + c];
        }
        fout << '\n';
    }
}

void store_valid_tile(std::vector<float>& compact,
                      const std::vector<float>& stream,
                      int sample_offset,
                      int tile_linear) {
    if (tile_linear >= srad_cfg::kTotalTileCount) return;

    const int tile_r = tile_linear / srad_cfg::kTileColCount;
    const int tile_c = tile_linear % srad_cfg::kTileColCount;
    const int tile_row_start = tile_r * srad_cfg::kTileStrideRows;
    const int tile_col_start = tile_c * srad_cfg::kTileStrideCols;

    for (int r = 0; r < srad_cfg::kOutputTileRows; ++r) {
        const int global_r = tile_row_start + r;
        if (global_r < srad_cfg::kOutputFirstRow ||
            global_r > srad_cfg::kOutputLastRow) {
            continue;
        }

        for (int c = 0; c < srad_cfg::kOutputTileCols; ++c) {
            const int global_c = tile_col_start + c;
            if (global_c < srad_cfg::kOutputFirstCol ||
                global_c > srad_cfg::kOutputLastCol) {
                continue;
            }

            const int out_r = global_r - srad_cfg::kOutputFirstRow;
            const int out_c = global_c - srad_cfg::kOutputFirstCol;
            compact[out_r * srad_cfg::kOutputCols + out_c] =
                stream[sample_offset + r * srad_cfg::kOutputTileCols + c];
        }
    }
}

} // namespace

int main() {
    std::printf("[aiesim] GraphOursPLQ0: %d lane(s), %d grouped output PLIO(s), %d graph firing(s) per lane, %d output floats per firing, two-kernel srad_local_q -> srad_coeff_update\n",
                srad_cfg::kParallelLanes,
                srad_cfg::kOutputPlioGroups,
                srad_cfg::kGraphRunIterations * srad_cfg::kSradIterations,
                srad_cfg::kOutputSampleElems);
    std::fflush(stdout);
    graphOursPLQ0.init();
    graphOursPLQ0.run(srad_cfg::kGraphRunIterations *
                      srad_cfg::kSradIterations);
    graphOursPLQ0.wait();
    graphOursPLQ0.end();

    std::vector<float> compact_out_buf(srad_cfg::kOutputElems, 0.0f);
    for (int group = 0; group < srad_cfg::kOutputPlioGroups; ++group) {
        std::vector<float> group_out_buf(
            srad_cfg::kAieOutputElemsPerLane *
            srad_cfg::kOutputLanesPerPlio);
        if (!load_float_file(srad_plio_files::j_next_out_group(group),
                             group_out_buf)) {
            return EXIT_FAILURE;
        }

        for (int srad_iter = 0;
             srad_iter < srad_cfg::kSradIterations;
             ++srad_iter) {
            for (int tile_iter = 0;
                 tile_iter < srad_cfg::kGraphRunIterations;
                 ++tile_iter) {
                const int graph_iter =
                    srad_iter * srad_cfg::kGraphRunIterations + tile_iter;
                for (int local = 0;
                     local < srad_cfg::kOutputLanesPerPlio;
                     ++local) {
                    if (srad_iter != srad_cfg::kSradIterations - 1) {
                        continue;
                    }

                    const int lane =
                        group * srad_cfg::kOutputLanesPerPlio + local;
                    const int tile_linear =
                        tile_iter * srad_cfg::kParallelLanes + lane;
                    const int sample_offset =
                        (graph_iter * srad_cfg::kOutputLanesPerPlio + local) *
                        srad_cfg::kOutputSampleElems;
                    store_valid_tile(compact_out_buf,
                                     group_out_buf,
                                     sample_offset,
                                     tile_linear);
                }
            }
        }
    }
    dump_float_file("./data/aiesim_j_next.txt", compact_out_buf.data());
    std::printf("[aiesim] wrote ./data/aiesim_j_next.txt\n");

    return EXIT_SUCCESS;
}
#endif
