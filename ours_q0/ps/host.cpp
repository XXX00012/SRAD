#include "TopGraph.h"
#include "ProcessUnit/include.h"

#include <adf/adf_api/XRTConfig.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>
#include <xrt/xrt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
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


using Clock = std::chrono::high_resolution_clock;

long long elapsed_us(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
}

int active_iterations(int iter_cnt) {
    int active = iter_cnt;
    if (active < 1) {
        active = 1;
    }
    if (active > srad_cfg::kBoardIterations) {
        active = srad_cfg::kBoardIterations;
    }
    return active;
}

bool load_input_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[error] cannot open %s\n", path.c_str());
        return false;
    }

    float value = 0.0f;
    int count = 0;
    while (count < static_cast<int>(buf.size()) && (fin >> value)) {
        buf[count++] = value;
    }

    if (count != static_cast<int>(buf.size())) {
        std::fprintf(stderr,
                     "[error] %s element count mismatch: got %d, expect %zu\n",
                     path.c_str(),
                     count,
                     buf.size());
        return false;
    }

    if (fin >> value) {
        std::fprintf(stderr,
                     "[warn] %s has extra elements after expected %zu; ignored\n",
                     path.c_str(),
                     buf.size());
    }

    return true;
}

void dump_output_matrix(const std::string& path, const float* output) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s for write\n", path.c_str());
        return;
    }

    for (int r = 0; r < srad_cfg::kBoardRows; ++r) {
        for (int c = 0; c < srad_cfg::kBoardCols; ++c) {
            if (c) {
                fout << ' ';
            }
            fout << output[r * srad_cfg::kBoardCols + c];
        }
        fout << '\n';
    }
}

xrt::kernel open_toppl_kernel(const xrt::device& device, const xrt::uuid& uuid) {
    try {
        return xrt::kernel(device, uuid, "TopPL:{TopPL_0}");
    } catch (const std::exception&) {
        try {
            return xrt::kernel(device, uuid, "TopPL_0");
        } catch (const std::exception&) {
            return xrt::kernel(device, uuid, "TopPL");
        }
    }
}

using Clock = std::chrono::high_resolution_clock;

constexpr int kDiagPollMs = 1000;

struct PipelineTiming {
    long long bo_to_device_us = 0;
    long long submit_us = 0;
    long long wait_all_us = 0;
    long long toppl_wait_us = 0;
    long long graph_wait_us = 0;
    long long bo_from_device_us = 0;
};


} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <xclbin> [iter_cnt] [input_txt] [output_txt]\n"
                     "       omit output_txt, or pass '-', for performance-only run\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string xclbin_path = argv[1];
    const int requested_iters =
        (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const int active_iters = active_iterations(requested_iters);
    const int graph_rows = srad_cfg::kBoardGraphRowsPerIteration * active_iters;
    const std::string input_path =
        (argc >= 4) ? argv[3] : "./data/input_image.txt";
    const bool dump_output = argc >= 5 && std::strcmp(argv[4], "-") != 0;
    const std::string output_path = dump_output ? argv[4] : std::string();

    std::printf("[host] image        : %d x %d\n",
                srad_cfg::kBoardRows,
                srad_cfg::kBoardCols);
    std::printf("[host] strips       : %d x %d cols, rows/strip=%d\n",
                srad_cfg::kBoardStrips,
                srad_cfg::kRowDataElems,
                srad_cfg::kBoardRowsPerStrip);
    std::printf("[host] requested it : %d\n", requested_iters);
    std::printf("[host] active it    : %d\n", active_iters);
    std::printf("[host] graph rows   : %d\n", graph_rows);
    std::printf("[host] output dump  : %s\n",
                dump_output ? output_path.c_str() : "skipped");

    std::vector<float> input(srad_cfg::kBoardPixels, 0.0f);
    if (!load_input_file(input_path, input)) {
        return EXIT_FAILURE;
    }

    bool graph_initialized = false;
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

        auto toppl = open_toppl_kernel(device, xrt_uuid);

        auto image_bo =
            xrt::bo(device, srad_cfg::kBoardImageBytes, toppl.group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, toppl.group_id(1));

        auto image_map = image_bo.map<float*>();
        auto output_map = output_bo.map<float*>();

        std::memcpy(image_map, input.data(), srad_cfg::kBoardImageBytes);

        PipelineTiming timing;
        const auto total_t0 = Clock::now();
        auto stage_t0 = Clock::now();
        image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        graphOursPLQ0.init();
        graph_initialized = true;

        graphOursPLQ0.run(graph_rows);
        const auto pl_t0 = Clock::now();
        auto toppl_run = toppl(image_bo, output_bo, active_iters);
        toppl_run.wait();
        const auto pl_t1 = Clock::now();
        graphOursPLQ0.wait();
        graphOursPLQ0.end();
      
        const auto pl_us =
            std::chrono::duration_cast<std::chrono::microseconds>(pl_t1 - pl_t0).count();

        std::printf("pl_total_us : %lld\n", static_cast<long long>(pl_us));    
       

        graph_initialized = false;

        if (dump_output) {
            stage_t0 = Clock::now();
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            timing.bo_from_device_us = elapsed_us(stage_t0, Clock::now());
            dump_output_matrix(output_path, output_map);
        }

        const long long total_us = elapsed_us(total_t0, Clock::now());

        std::printf("timing us: h2d=%lld submit=%lld ddr_to_ddr_kernel=%lld toppl_wait=%lld graph_wait=%lld d2h=%lld total=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.wait_all_us,
                    timing.toppl_wait_us,
                    timing.graph_wait_us,
                    timing.bo_from_device_us,
                    total_us);
        std::printf("ddr_to_ddr_kernel_us: %lld\n", timing.wait_all_us);
        if (dump_output) {
            std::printf("[host] output       : %s\n", output_path.c_str());
        }

        xrtDeviceClose(dhdl);
        dhdl = nullptr;
    } catch (const std::exception& ex) {
        if (graph_initialized) {
            try {
                graphOursPLQ0.end();
            } catch (...) {
            }
        }
        if (dhdl) {
            xrtDeviceClose(dhdl);
        }
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
