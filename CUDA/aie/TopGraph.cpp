#include "TopGraph.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <string>
#include <vector>

GraphPrepare graphPrepare("prepare");
GraphReduce graphReduce("reduce");
GraphCoeff graphCoeff("coeff");
GraphUpdate graphUpdate("update");

#if defined(__AIESIM__) || defined(__X86SIM__)
namespace {

std::string normalized_relative_path(const char* path) {
    std::string rel(path);
    while (rel.rfind("./", 0) == 0) {
        rel.erase(0, 2);
    }
    return rel;
}

bool file_exists(const std::string& path) {
    std::ifstream fin(path);
    return fin.is_open();
}

bool plio_file_exists(const char* path) {
    return file_exists(normalized_relative_path(path));
}

bool plio_files_exist(std::initializer_list<const char*> paths) {
    for (const char* path : paths) {
        if (!plio_file_exists(path)) {
            return false;
        }
    }
    return true;
}

bool load_stream_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[aiesim] cannot open %s\n", path.c_str());
        return false;
    }

    float v = 0.0f;
    int n = 0;
    while (fin >> v) {
        if (n >= static_cast<int>(buf.size())) {
            break;
        }
        buf[n++] = v;
    }

    if (n != static_cast<int>(buf.size())) {
        std::fprintf(stderr,
                     "[aiesim] %s element count mismatch: got %d, expect %zu\n",
                     path.c_str(),
                     n,
                     buf.size());
        return false;
    }
    return true;
}

bool load_generated_stream(const char* plio_path, std::vector<float>& buf) {
    const std::string rel = normalized_relative_path(plio_path);
    const std::string aie_path = std::string("aiesimulator_output/") + rel;
    const std::string x86_path = std::string("x86simulator_output/") + rel;

    if (file_exists(aie_path)) {
        return load_stream_file(aie_path, buf);
    }
    if (file_exists(x86_path)) {
        return load_stream_file(x86_path, buf);
    }

    std::fprintf(stderr,
                 "[aiesim] cannot find generated PLIO output for %s "
                 "(checked %s and %s)\n",
                 plio_path,
                 aie_path.c_str(),
                 x86_path.c_str());
    return false;
}

bool write_stream(const char* path, const std::vector<float>& buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr, "[aiesim] cannot open %s for write\n", path);
        return false;
    }

    fout << std::setprecision(9);
    for (float v : buf) {
        fout << v << '\n';
    }
    return true;
}

float compute_q0sqr_from_stats(float sum, float sum2) {
    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float mean2 = mean * mean;
    const float var = (sum2 / static_cast<float>(srad_cfg::kPixels)) - mean2;
    return var / mean2;
}

std::vector<float> scalar_packets(float value) {
    std::vector<float> out(srad_cfg::kTileCount * srad_cfg::kScalarPacketElems,
                           0.0f);
    for (int tile = 0; tile < srad_cfg::kTileCount; ++tile) {
        out[tile * srad_cfg::kScalarPacketElems] = value;
    }
    return out;
}

int clamp_index(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

int opencl_index(int row, int col) {
    return row + srad_cfg::kRows * col;
}

void unpack_plain_tiles(const std::vector<float>& tiles,
                        std::vector<float>& image) {
    int pos = 0;
    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            const int row0 = tr * srad_cfg::kBlockRows;
            const int col0 = tc * srad_cfg::kBlockCols;
            const int rows =
                std::min(srad_cfg::kBlockRows, srad_cfg::kRows - row0);
            const int cols =
                std::min(srad_cfg::kBlockCols, srad_cfg::kCols - col0);

            for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
                for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                    const float v = tiles[pos++];
                    if (r < rows && c < cols) {
                        image[opencl_index(row0 + r, col0 + c)] = v;
                    }
                }
            }
        }
    }
}

std::vector<float> pack_plain_tiles(const std::vector<float>& image) {
    std::vector<float> out;
    out.reserve(srad_cfg::kTileCount * srad_cfg::kBlockPixels);

    for (int tr = 0; tr < srad_cfg::kTileRows; ++tr) {
        for (int tc = 0; tc < srad_cfg::kTileCols; ++tc) {
            const int row0 = tr * srad_cfg::kBlockRows;
            const int col0 = tc * srad_cfg::kBlockCols;
            for (int c = 0; c < srad_cfg::kBlockCols; ++c) {
                for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                    const int gr =
                        clamp_index(row0 + r, 0, srad_cfg::kRows - 1);
                    const int gc =
                        clamp_index(col0 + c, 0, srad_cfg::kCols - 1);
                    out.push_back(image[opencl_index(gr, gc)]);
                }
            }
        }
    }

    return out;
}

std::vector<float> pack_update_c_tiles(const std::vector<float>& c_plane) {
    std::vector<float> out;
    out.reserve(srad_cfg::kTileCount * srad_cfg::kUpdateCInputElems);

    for (int tile = 0; tile < srad_cfg::kTileCount; ++tile) {
        out.insert(out.end(), c_plane.begin(), c_plane.end());
    }

    return out;
}

bool prepare_coeff_inputs_from_reduce() {
    std::vector<float> stats(srad_cfg::kScalarPacketElems, 0.0f);
    if (!load_generated_stream(srad_plio_files::kReduceStatsOut, stats)) {
        if (plio_file_exists(srad_plio_files::kCoeffQ0)) {
            std::printf("[aiesim] using pre-generated %s\n",
                        srad_plio_files::kCoeffQ0);
            return true;
        }
        return false;
    }

    const float q0sqr = compute_q0sqr_from_stats(stats[0], stats[1]);
    std::printf("[aiesim] reduce_sum=%.9g reduce_sum2=%.9g q0sqr=%.9g\n",
                stats[0],
                stats[1],
                q0sqr);
    return write_stream(srad_plio_files::kCoeffQ0, scalar_packets(q0sqr));
}

bool prepare_reduce_inputs_from_prepare() {
    std::vector<float> sums(srad_cfg::kPixels, 0.0f);
    std::vector<float> sums2(srad_cfg::kPixels, 0.0f);

    if (!load_generated_stream(srad_plio_files::kPrepareSumsOut, sums) ||
        !load_generated_stream(srad_plio_files::kPrepareSums2Out, sums2)) {
        if (plio_files_exist({srad_plio_files::kReduceSums,
                              srad_plio_files::kReduceSums2})) {
            std::printf("[aiesim] using pre-generated prepare outputs for reduce inputs\n");
            return true;
        }
        return false;
    }

    return write_stream(srad_plio_files::kReduceSums, sums) &&
           write_stream(srad_plio_files::kReduceSums2, sums2);
}

bool prepare_update_inputs_from_coeff() {
    const int tile_output_elems =
        srad_cfg::kTileCount * srad_cfg::kBlockPixels;
    std::vector<float> c_tiles(tile_output_elems);
    std::vector<float> dN_tiles(tile_output_elems);
    std::vector<float> dS_tiles(tile_output_elems);
    std::vector<float> dW_tiles(tile_output_elems);
    std::vector<float> dE_tiles(tile_output_elems);

    if (!load_generated_stream(srad_plio_files::kCoeffCOut, c_tiles) ||
        !load_generated_stream(srad_plio_files::kCoeffDNOut, dN_tiles) ||
        !load_generated_stream(srad_plio_files::kCoeffDSOut, dS_tiles) ||
        !load_generated_stream(srad_plio_files::kCoeffDWOut, dW_tiles) ||
        !load_generated_stream(srad_plio_files::kCoeffDEOut, dE_tiles)) {
        if (plio_files_exist({srad_plio_files::kUpdateC,
                              srad_plio_files::kUpdateDN,
                              srad_plio_files::kUpdateDS,
                              srad_plio_files::kUpdateDW,
                              srad_plio_files::kUpdateDE,
                              srad_plio_files::kUpdateLambda})) {
            std::printf("[aiesim] using pre-generated coeff outputs for update inputs\n");
            return true;
        }
        return false;
    }

    std::vector<float> c_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dN_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dS_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dW_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dE_plane(srad_cfg::kPixels, 0.0f);
    unpack_plain_tiles(c_tiles, c_plane);
    unpack_plain_tiles(dN_tiles, dN_plane);
    unpack_plain_tiles(dS_tiles, dS_plane);
    unpack_plain_tiles(dW_tiles, dW_plane);
    unpack_plain_tiles(dE_tiles, dE_plane);

    return write_stream(srad_plio_files::kUpdateC,
                        pack_update_c_tiles(c_plane)) &&
           write_stream(srad_plio_files::kUpdateDN,
                        pack_plain_tiles(dN_plane)) &&
           write_stream(srad_plio_files::kUpdateDS,
                        pack_plain_tiles(dS_plane)) &&
           write_stream(srad_plio_files::kUpdateDW,
                        pack_plain_tiles(dW_plane)) &&
           write_stream(srad_plio_files::kUpdateDE,
                        pack_plain_tiles(dE_plane)) &&
           write_stream(srad_plio_files::kUpdateLambda,
                        scalar_packets(srad_cfg::kLambdaDefault));
}

int run_graphs() {
    graphPrepare.init();
    graphPrepare.run(1);
    graphPrepare.end();

    if (!prepare_reduce_inputs_from_prepare()) {
        return EXIT_FAILURE;
    }

    graphReduce.init();
    graphReduce.run(1);
    graphReduce.end();

    if (!prepare_coeff_inputs_from_reduce()) {
        return EXIT_FAILURE;
    }

    graphCoeff.init();
    graphCoeff.run(srad_cfg::kTileCount);
    graphCoeff.end();

    if (!prepare_update_inputs_from_coeff()) {
        return EXIT_FAILURE;
    }

    graphUpdate.init();
    graphUpdate.run(srad_cfg::kTileCount);
    graphUpdate.end();

    return 0;
}

} // namespace

#if defined(__PS_INIT_AIE__)
int ps_main() {
    return run_graphs();
}
#else
int main() {
    return run_graphs();
}
#endif
#endif
