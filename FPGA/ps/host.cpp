// Board wrapper for the existing srad_fpga_v5 AIE kernel:
//   DDR -> LoadFpgaV5 -> GraphFpgaV5(srad_fpga_v5) -> StoreFpgaV5 -> DDR.
//
// The AIE kernel computes q0sqr internally and emits sparse (index,value)
// updates. PL only marshals DDR data into the stream order expected by AIE.

#include "TopGraph.h"
#include "ProcessUnit/include.h"

#include <adf/adf_api/XRTConfig.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

GraphFpgaV5 graphFpgaV5("fpga_v5");

namespace {

constexpr int kPreview = 16;
constexpr float kCompareTol = 1.0e-5f;
constexpr int kCpuReferenceMaxPixels = 1024 * 1024;
using Clock = std::chrono::high_resolution_clock;

struct ReferenceData {
    float q0sqr = 0.0f;
    std::vector<float> j_next;
};

void log_stage(const char* message) {
    std::printf("[stage] %s\n", message);
    std::fflush(stdout);
}

std::string default_input_path() {
    return std::string("./data/input_") + std::to_string(ROW) + "x" +
           std::to_string(COL) + ".txt";
}

bool load_float_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s\n", path.c_str());
        return false;
    }

    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            float value = 0.0f;
            if (!(fin >> value)) {
                std::fprintf(stderr,
                             "[warn] %s element count mismatch, expect %d values\n",
                             path.c_str(),
                             srad_cfg::kPixels);
                return false;
            }
            buf[srad_math::image_index(row, col)] = value;
        }
    }
    return true;
}

void fill_default_image(std::vector<float>& image) {
    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            image[srad_math::image_index(row, col)] =
                1.0f + 0.003f * static_cast<float>(row) +
                0.002f * static_cast<float>(col) +
                0.05f * std::sin(0.31f * static_cast<float>(row)) *
                    std::cos(0.19f * static_cast<float>(col));
        }
    }
}

void dump_float_file(const std::string& path, const std::vector<float>& buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s for write\n", path.c_str());
        return;
    }

    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            if (col) {
                fout << ' ';
            }
            fout << buf[srad_math::image_index(row, col)];
        }
        fout << '\n';
    }
}

void print_preview(const char* tag, const std::vector<float>& buf) {
    std::printf("%s", tag);
    const int n = std::min<int>(kPreview, buf.size());
    for (int i = 0; i < n; ++i) {
        std::printf(" %.9g", buf[i]);
    }
    std::printf("\n");
}

float compute_q0sqr_reference(const std::vector<float>& image) {
    float sum = 0.0f;
    float sum2 = 0.0f;

    for (float value : image) {
        sum += value;
        sum2 += value * value;
    }

    const float mean = sum / static_cast<float>(srad_cfg::kPixels);
    const float variance =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - (mean * mean);
    return (mean != 0.0f) ? (variance / (mean * mean)) : 0.0f;
}

float compute_c_reference(float jc,
                          float dN,
                          float dS,
                          float dW,
                          float dE,
                          float q0sqr) {
    if (srad_cfg::kBypassCoeffMath) {
        return 1.0f;
    }

    const float g2 =
        (dN * dN + dS * dS + dW * dW + dE * dE) / (jc * jc);
    const float lap = (dN + dS + dW + dE) / jc;
    const float num = 0.5f * g2 - (1.0f / 16.0f) * lap * lap;
    float den = 1.0f + 0.25f * lap;
    const float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    return srad_math::clamp01(1.0f / (1.0f + den));
}

ReferenceData cpu_reference(const std::vector<float>& image, float lambda) {
    ReferenceData ref;
    ref.q0sqr = compute_q0sqr_reference(image);
    ref.j_next.assign(srad_cfg::kPixels, 0.0f);

    std::vector<float> coeff(srad_cfg::kPixels, 0.0f);

    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            const int p = srad_math::image_index(row, col);
            const float jc = image[p];
            const float dN =
                image[srad_math::image_index(srad_math::north_row(row), col)] - jc;
            const float dS =
                image[srad_math::image_index(srad_math::south_row(row), col)] - jc;
            const float dW =
                image[srad_math::image_index(row, srad_math::west_col(col))] - jc;
            const float dE =
                image[srad_math::image_index(row, srad_math::east_col(col))] - jc;
            coeff[p] = compute_c_reference(jc, dN, dS, dW, dE, ref.q0sqr);
        }
    }

    for (int row = 0; row < ROW; ++row) {
        for (int col = 0; col < COL; ++col) {
            const int p = srad_math::image_index(row, col);
            const int south = srad_math::image_index(srad_math::south_row(row), col);
            const int east = srad_math::image_index(row, srad_math::east_col(col));
            const float jc = image[p];
            const float dN =
                image[srad_math::image_index(srad_math::north_row(row), col)] - jc;
            const float dS = image[south] - jc;
            const float dW =
                image[srad_math::image_index(row, srad_math::west_col(col))] - jc;
            const float dE = image[east] - jc;
            const float d =
                coeff[p] * dN + coeff[south] * dS + coeff[p] * dW + coeff[east] * dE;
            ref.j_next[p] = jc + 0.25f * lambda * d;
        }
    }

    return ref;
}

void compare_output(const std::vector<float>& got,
                    const std::vector<float>& gold) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int mismatch_count = 0;

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
        const float abs_err = std::fabs(got[i] - gold[i]);
        const float denom = std::max(std::fabs(gold[i]), 1.0e-12f);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, abs_err / denom);
        if (abs_err > kCompareTol) {
            ++mismatch_count;
        }
    }

    std::printf("max_abs_error_float : %.9g\n", max_abs);
    std::printf("max_relative_error  : %.9g\n", max_rel);
    std::printf("mismatch_count_tol_%g: %d\n", kCompareTol, mismatch_count);
}

xrt::kernel open_kernel_or_die(const xrt::device& device,
                               const xrt::uuid& uuid,
                               const char* name) {
    try {
        const std::string cu_name =
            std::string(name) + ":{" + name + "_1}";
        return xrt::kernel(device, uuid, cu_name.c_str());
    } catch (const std::exception&) {
        return xrt::kernel(device, uuid, name);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <xclbin> [iter_cnt] [input_txt] [output_txt] [lambda]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string xclbin_path = argv[1];
    const int iter_cnt =
        (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const std::string input_path =
        (argc >= 4) ? argv[3] : default_input_path();
    const std::string output_path =
        (argc >= 5) ? argv[4] : "./data/aie_j_next.txt";
    const float lambda = (argc >= 6) ? static_cast<float>(std::atof(argv[5]))
                                     : srad_cfg::kLambdaDefault;

    if (iter_cnt <= 0) {
        std::fprintf(stderr,
                     "[error] iteration count must be positive; got %d\n",
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels, 0.0f);
    if (!load_float_file(input_path, image)) {
        std::fprintf(stderr, "[warn] fallback to deterministic input\n");
        fill_default_image(image);
    }

    const bool run_reference =
        (iter_cnt == 1) && (srad_cfg::kPixels <= kCpuReferenceMaxPixels);
    ReferenceData ref;
    if (run_reference) {
        ref = cpu_reference(image, lambda);
    }

    std::printf("image size            : %dx%d (%d pixels)\n",
                ROW,
                COL,
                srad_cfg::kPixels);
    std::printf("iterations requested  : %d\n", iter_cnt);
    std::printf("tile geometry         : %dx%d tiles, input tile %dx%d\n",
                srad_cfg::kTileRows,
                srad_cfg::kTileCols,
                srad_cfg::kInputTileRows,
                srad_cfg::kInputTileCols);
    std::printf("tiles per iteration   : %d\n",
                srad_cfg::kTilesPerIteration);
    std::printf("compute stream floats : %d\n",
                srad_cfg::kComputeStreamElems);
    std::printf("output stream floats  : %d\n",
                srad_cfg::kOutputStreamElems);
    if (run_reference) {
        std::printf("q0sqr_ref float32     : %.9g\n", ref.q0sqr);
    } else {
        std::printf("reference compare     : skipped");
        if (iter_cnt != 1) {
            std::printf(" (multi-iteration run)");
        } else {
            std::printf(" (image exceeds %d pixels)", kCpuReferenceMaxPixels);
        }
        std::printf("\n");
    }
    std::printf("lambda float32        : %.9g\n", lambda);
    print_preview("input preview:", image);

    bool graph_needs_end = false;
    bool graph_end_attempted = false;

    try {
        log_stage("open XRT device 0");
        auto device = xrt::device(0);

        log_stage("load xclbin");
        auto xrt_uuid = device.load_xclbin(xclbin_path);

        log_stage("open XRT C handle from loaded xclbin");
        xrtDeviceHandle dhdl = xrtDeviceOpenFromXcl(device);
        if (!dhdl) {
            std::fprintf(stderr, "[error] xrtDeviceOpenFromXcl failed\n");
            return EXIT_FAILURE;
        }

        log_stage("register ADF graph with XRT");
        adf::registerXRT(dhdl, xrt_uuid.get());

        log_stage("open PL kernels");
        auto load_fpga = open_kernel_or_die(device, xrt_uuid, "LoadFpgaV5");
        auto store_fpga = open_kernel_or_die(device, xrt_uuid, "StoreFpgaV5");

        log_stage("allocate BOs");
        auto image_bo =
            xrt::bo(device, srad_cfg::kImageBytes, load_fpga.group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kOutputBytes, store_fpga.group_id(2));

        log_stage("map BOs");
        auto image_map = image_bo.map<float*>();
        auto output_map = output_bo.map<float*>();

        log_stage("prepare BO contents");
        std::memcpy(image_map, image.data(), srad_cfg::kImageBytes);
        std::memset(output_map, 0, srad_cfg::kOutputBytes);

        log_stage("sync input BO to device");
        image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        log_stage("graph init");
        graphFpgaV5.init();
        graph_needs_end = true;

        long long pl_us = 0;
        const auto pipeline_t0 = Clock::now();
        for (int iter = 0; iter < iter_cnt; ++iter) {
            const int iter_id = iter + 1;
            std::printf("[stage] iteration %d/%d run PL/AIE transaction\n",
                        iter_id,
                        iter_cnt);
            std::fflush(stdout);

            const auto iter_pl_t0 = Clock::now();
            auto sink_run = store_fpga(nullptr, nullptr, output_bo);
            graphFpgaV5.run(1);
            auto source_run = load_fpga(image_bo, lambda, nullptr, nullptr);
            source_run.wait();
            sink_run.wait();
            const auto iter_pl_t1 = Clock::now();
            pl_us += std::chrono::duration_cast<std::chrono::microseconds>(
                         iter_pl_t1 - iter_pl_t0)
                         .count();

            std::printf("[stage] iteration %d/%d wait graph\n",
                        iter_id,
                        iter_cnt);
            std::fflush(stdout);
            graphFpgaV5.wait();

            log_stage("sync output BO from device");
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (iter + 1 < iter_cnt) {
                log_stage("prepare next iteration input");
                std::memcpy(image_map, output_map, srad_cfg::kImageBytes);
                image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
        }
        const auto pipeline_t1 = Clock::now();
        const auto pipeline_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                pipeline_t1 - pipeline_t0)
                .count();

        log_stage("graph end");
        graph_end_attempted = true;
        graphFpgaV5.end();
        graph_needs_end = false;

        log_stage("process output");
        std::vector<float> got(output_map,
                               output_map + srad_cfg::kOutputElems);
        print_preview("output preview:", got);
        if (run_reference) {
            compare_output(got, ref.j_next);
        } else {
            std::printf("reference compare     : skipped\n");
        }
        dump_float_file(output_path, got);

        std::printf("pl_total_us : %lld\n", static_cast<long long>(pl_us));
        std::printf("pipeline_total_us : %lld\n",
                    static_cast<long long>(pipeline_us));

        log_stage("close XRT device");
        xrtDeviceClose(dhdl);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        std::fflush(stderr);
        if (graph_needs_end && !graph_end_attempted) {
            try {
                std::fprintf(stderr, "[cleanup] graph end after exception\n");
                std::fflush(stderr);
                graph_end_attempted = true;
                graphFpgaV5.end();
                graph_needs_end = false;
            } catch (const std::exception& cleanup_ex) {
                std::fprintf(stderr,
                             "[cleanup] graph end failed after exception: %s\n",
                             cleanup_ex.what());
                std::fflush(stderr);
            } catch (...) {
                std::fprintf(stderr,
                             "[cleanup] graph end failed after exception\n");
                std::fflush(stderr);
            }
        }
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
