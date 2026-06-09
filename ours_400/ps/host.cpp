// Ours400 board runner:
//   10 TopPL CUs feed 200 AIE lanes.
//   Q0Ctrl computes q0sqr from per-worker partial stats.
//   AIE K1/K2 run SRAD tile updates and PL writes the final image to DDR.
//
// This host is intentionally performance-only: no CPU reference, no polling
// sleep, no output D2H, and no file dump. The primary metric is the wait phase,
// which covers DDR read -> AIE compute -> DDR write inside the accelerator.

#include "TopGraph.h"

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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

GraphOursPLQ0 graphOursPLQ0("ours_plq0");

namespace {

constexpr int PL_WORKERS = srad_cfg::kTopPlWorkers;
constexpr int TOPPL_DEBUG_BASE = srad_cfg::kOutputElems;
constexpr int Q0_DEBUG_SLOTS = PL_WORKERS + 2;
constexpr int DEBUG_FLOATS = PL_WORKERS + Q0_DEBUG_SLOTS;
constexpr int OUTPUT_BO_FLOATS = srad_cfg::kOutputElems + DEBUG_FLOATS;
constexpr int OUTPUT_BO_BYTES = OUTPUT_BO_FLOATS * srad_cfg::kScalarBytes;

using Clock = std::chrono::high_resolution_clock;

struct PipelineTiming {
    long long bo_to_device_us = 0;
    long long submit_us = 0;
    long long wait_all_us = 0;
    long long toppl_wait_us = 0;
    long long q0_wait_us = 0;
    long long graph_wait_us = 0;
};

long long elapsed_us(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
}

bool load_float_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s\n", path.c_str());
        return false;
    }

    float value = 0.0f;
    int count = 0;
    while (fin >> value) {
        if (count >= static_cast<int>(buf.size())) {
            break;
        }
        buf[count++] = value;
    }

    if (count != static_cast<int>(buf.size())) {
        std::fprintf(stderr,
                     "[warn] %s element count mismatch: got %d, expect %zu\n",
                     path.c_str(),
                     count,
                     buf.size());
        return false;
    }
    return true;
}

void fill_default_image(std::vector<float>& image) {
    for (int r = 0; r < srad_cfg::kRows; ++r) {
        for (int c = 0; c < srad_cfg::kCols; ++c) {
            image[r * srad_cfg::kCols + c] =
                1.0f + 0.003f * static_cast<float>(r) +
                0.002f * static_cast<float>(c) +
                0.05f * std::sin(0.31f * static_cast<float>(r)) *
                    std::cos(0.19f * static_cast<float>(c));
        }
    }
}

xrt::kernel open_toppl_kernel(const xrt::device& device,
                              const xrt::uuid& uuid,
                              int cu_index) {
    char scoped[80];
    std::snprintf(scoped,
                  sizeof(scoped),
                  "TopPL:{TopPL_%d}",
                  cu_index);
    std::printf("[open] %s\n", scoped);
    std::fflush(stdout);
    return xrt::kernel(device, uuid, scoped);
}

xrt::kernel open_q0ctrl_kernel(const xrt::device& device,
                               const xrt::uuid& uuid) {
    std::printf("[open] Q0Ctrl:{Q0Ctrl_1}\n");
    std::fflush(stdout);
    return xrt::kernel(device, uuid, "Q0Ctrl:{Q0Ctrl_1}");
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <xclbin> [iter_cnt] [input_txt]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string xclbin_path = argv[1];
    const int iter_cnt =
        (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const std::string input_path =
        (argc >= 4) ? argv[3] : "./data/plio_ours_j.txt";

    if (iter_cnt < 1 || iter_cnt > srad_cfg::kSradIterations) {
        std::fprintf(stderr,
                     "[error] Ours400 supports 1..%d iteration(s); got %d\n",
                     srad_cfg::kSradIterations,
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels);
    if (!load_float_file(input_path, image)) {
        std::fprintf(stderr, "[warn] fallback to deterministic input\n");
        fill_default_image(image);
    }

    std::printf("mapping               : %d x TopPL + Q0Ctrl + %d-lane AIE K1/K2 (%d lane(s)/TopPL)\n",
                srad_cfg::kTopPlWorkers,
                srad_cfg::kParallelLanes,
                srad_cfg::kWorkerLanes);
    std::printf("iterations            : %d\n", iter_cnt);
    std::printf("image size            : %dx%d (%d pixels)\n",
                srad_cfg::kRows,
                srad_cfg::kCols,
                srad_cfg::kPixels);
    std::printf("tile schedule         : %dx%d tiles, graph.run(%d), %d lanes, output groups=%d pktmerge<%d>\n",
                srad_cfg::kTileRowCount,
                srad_cfg::kTileColCount,
                srad_cfg::kGraphRunIterations * iter_cnt,
                srad_cfg::kParallelLanes,
                srad_cfg::kOutputPlioGroups,
                srad_cfg::kOutputLanesPerPlio);
    std::printf("data size bytes       : J=%d J_next=%d output_bo_total=%d\n",
                srad_cfg::kImageBytes,
                srad_cfg::kOutputBytes,
                OUTPUT_BO_BYTES);
    std::fflush(stdout);

    xrtDeviceHandle dhdl = nullptr;

    try {
        auto device = xrt::device(0);
        auto xrt_uuid = device.load_xclbin(xclbin_path);

        dhdl = xrtDeviceOpenFromXcl(device);
        if (!dhdl) {
            std::fprintf(stderr, "[error] xrtDeviceOpenFromXcl failed\n");
            return EXIT_FAILURE;
        }

        xuid_t adf_uuid;
        xrtDeviceGetXclbinUUID(dhdl, adf_uuid);
        adf::registerXRT(dhdl, adf_uuid);

        std::vector<xrt::kernel> toppl_kernels;
        toppl_kernels.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_kernels.emplace_back(
                open_toppl_kernel(device, xrt_uuid, i));
        }
        auto q0_kernel = open_q0ctrl_kernel(device, xrt_uuid);

        auto input_bo =
            xrt::bo(device, srad_cfg::kImageBytes, toppl_kernels[0].group_id(0));
        auto output_bo =
            xrt::bo(device, OUTPUT_BO_BYTES, toppl_kernels[0].group_id(1));
        auto input_map = input_bo.map<float*>();
        auto output_map = output_bo.map<float*>();

        std::memcpy(input_map, image.data(), srad_cfg::kImageBytes);
        std::memset(output_map, 0, OUTPUT_BO_BYTES);

        PipelineTiming timing;
        const auto total_t0 = Clock::now();

        auto stage_t0 = Clock::now();
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        graphOursPLQ0.init();

        stage_t0 = Clock::now();
        auto q0_run = q0_kernel(output_bo, iter_cnt);

        std::vector<xrt::run> toppl_runs;
        toppl_runs.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_runs.emplace_back(toppl_kernels[i](input_bo,
                                                     output_bo,
                                                     iter_cnt,
                                                     i));
        }

        graphOursPLQ0.run(srad_cfg::kGraphRunIterations * iter_cnt);
        timing.submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        std::vector<long long> toppl_wait_elapsed(PL_WORKERS, 0);
        std::vector<std::exception_ptr> toppl_errors(PL_WORKERS);

        std::vector<std::thread> toppl_wait_threads;
        toppl_wait_threads.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_wait_threads.emplace_back([&, i]() {
                const auto wait_t0 = Clock::now();
                try {
                    toppl_runs[i].wait();
                } catch (...) {
                    toppl_errors[i] = std::current_exception();
                }
                toppl_wait_elapsed[i] = elapsed_us(wait_t0, Clock::now());
            });
        }

        long long q0_wait_elapsed = 0;
        std::exception_ptr q0_error;
        std::thread q0_wait_thread([&]() {
            const auto wait_t0 = Clock::now();
            try {
                q0_run.wait();
            } catch (...) {
                q0_error = std::current_exception();
            }
            q0_wait_elapsed = elapsed_us(wait_t0, Clock::now());
        });

        long long graph_wait_elapsed = 0;
        std::exception_ptr graph_error;
        std::thread graph_wait_thread([&]() {
            const auto wait_t0 = Clock::now();
            try {
                graphOursPLQ0.wait();
            } catch (...) {
                graph_error = std::current_exception();
            }
            graph_wait_elapsed = elapsed_us(wait_t0, Clock::now());
        });

        for (auto& thread : toppl_wait_threads) {
            thread.join();
        }
        q0_wait_thread.join();
        graph_wait_thread.join();

        timing.wait_all_us = elapsed_us(stage_t0, Clock::now());
        timing.q0_wait_us = q0_wait_elapsed;
        timing.graph_wait_us = graph_wait_elapsed;
        for (long long value : toppl_wait_elapsed) {
            timing.toppl_wait_us = std::max(timing.toppl_wait_us, value);
        }

        if (q0_error) {
            std::rethrow_exception(q0_error);
        }
        if (graph_error) {
            std::rethrow_exception(graph_error);
        }
        for (int i = 0; i < PL_WORKERS; ++i) {
            if (toppl_errors[i]) {
                std::rethrow_exception(toppl_errors[i]);
            }
        }

        graphOursPLQ0.end();

        const long long total_us = elapsed_us(total_t0, Clock::now());

        std::printf("timing us: h2d=%lld submit=%lld ddr_to_ddr_kernel=%lld toppl_max=%lld q0_wait=%lld graph_wait=%lld total_no_d2h=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.wait_all_us,
                    timing.toppl_wait_us,
                    timing.q0_wait_us,
                    timing.graph_wait_us,
                    total_us);
        std::printf("ddr_to_ddr_kernel_us: %lld\n", timing.wait_all_us);
        std::fflush(stdout);

        xrtDeviceClose(dhdl);
        dhdl = nullptr;
    } catch (const std::exception& ex) {
        if (dhdl) {
            xrtDeviceClose(dhdl);
        }
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
