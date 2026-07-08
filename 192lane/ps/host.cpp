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

xrt::kernel open_kernel_cu(const xrt::device& device,
                           const xrt::uuid& uuid,
                           const char* kernel_name,
                           const char* cu_name) {
    const std::string scoped =
        std::string(kernel_name) + ":{" + cu_name + "}";
    try {
        return xrt::kernel(device, uuid, scoped.c_str());
    } catch (const std::exception&) {
        try {
            return xrt::kernel(device, uuid, cu_name);
        } catch (const std::exception&) {
            return xrt::kernel(device, uuid, kernel_name);
        }
    }
}

constexpr int kDiagPollMs = 1000;
constexpr const char* kBuildTag = "ours_192lane-12toppl-pktmerge-q0ctrl-20260626";

void log_stage(const char* stage) {
    std::fprintf(stderr, "[diag] %s\n", stage);
    std::fflush(stderr);
    std::fflush(stdout);
}

void log_value(const char* name, const std::string& value) {
    std::fprintf(stderr, "[diag] %s: %s\n", name, value.c_str());
    std::fflush(stderr);
    std::fflush(stdout);
}

struct PipelineTiming {
    long long bo_to_device_us = 0;
    long long submit_us = 0;
    long long wait_all_us = 0;
    long long graph_init_us = 0;
    long long graph_run_submit_us = 0;
    long long q0_launch_us = 0;
    long long q0_wait_us = 0;
    long long toppl_launch_us = 0;
    long long toppl_wait_us = 0;
    long long graph_wait_us = 0;
    long long graph_end_us = 0;
    long long bo_from_device_us = 0;
};


} // namespace

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    log_stage(kBuildTag);

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

    log_value("xclbin", xclbin_path);
    log_value("input", input_path);
    if (dump_output) {
        log_value("output", output_path);
    } else {
        log_stage("output: skipped");
    }

    std::printf("[host] image        : %d x %d\n",
                srad_cfg::kBoardRows,
                srad_cfg::kBoardCols);
    std::printf("[host] strips       : %d x %d cols, stream rows/lane=%d\n",
                srad_cfg::kBoardStrips,
                srad_cfg::kRowDataElems,
                srad_cfg::kBoardRowsPerStrip);
    std::printf("[host] lanes        : %d AIE lanes, %d TopPL workers x %d lanes\n",
                srad_cfg::kParallelLanes,
                srad_cfg::kTopPlWorkers,
                srad_cfg::kLanesPerTopPl);
    std::printf("[host] worker cols  : %d contiguous cols/TopPL\n",
                srad_cfg::kWorkerCols);
    std::printf("[host] row blocks   : %d x %d center rows, padded rows=%d\n",
                srad_cfg::kBoardRowBlocks,
                srad_cfg::kRowsPerRowBlock,
                srad_cfg::kBoardPaddedRows);
    std::printf("[host] requested it : %d\n", requested_iters);
    std::printf("[host] active it    : %d\n", active_iters);
    std::printf("[host] graph rows   : %d\n", graph_rows);
    std::printf("[host] output dump  : %s\n",
                dump_output ? output_path.c_str() : "skipped");

    std::vector<float> input(srad_cfg::kBoardPixels, 0.0f);
    log_stage("before load_input_file");
    if (!load_input_file(input_path, input)) {
        return EXIT_FAILURE;
    }
    log_stage("after load_input_file");

    bool graph_initialized = false;
    xrtDeviceHandle dhdl = nullptr;

    try {
        log_stage("before xrt::device(0)");
        auto device = xrt::device(0);
        log_stage("after xrt::device(0)");

        log_stage("before device.load_xclbin");
        auto xrt_uuid = device.load_xclbin(xclbin_path);
        log_stage("after device.load_xclbin");

        log_stage("before xrtDeviceOpenFromXcl");
        dhdl = xrtDeviceOpenFromXcl(device);
        if (!dhdl) {
            std::fprintf(stderr, "[error] xrtDeviceOpenFromXcl failed\n");
            return EXIT_FAILURE;
        }
        log_stage("after xrtDeviceOpenFromXcl");

        xuid_t adf_uuid;
        log_stage("before xrtDeviceGetXclbinUUID");
        xrtDeviceGetXclbinUUID(dhdl, adf_uuid);
        log_stage("after xrtDeviceGetXclbinUUID");

        log_stage("before adf::registerXRT");
        adf::registerXRT(dhdl, adf_uuid);
        log_stage("after adf::registerXRT");

        log_stage("before open PL kernels");
        std::vector<xrt::kernel> toppl;
        toppl.reserve(srad_cfg::kTopPlWorkers);
        for (int worker = 0; worker < srad_cfg::kTopPlWorkers; ++worker) {
            const std::string cu_name =
                "TopPL_" + std::to_string(worker);
            toppl.push_back(open_kernel_cu(device,
                                           xrt_uuid,
                                           "TopPL",
                                           cu_name.c_str()));
        }
        auto q0ctrl = open_kernel_cu(device, xrt_uuid, "Q0Ctrl", "Q0Ctrl_0");
        log_stage("after open PL kernels");

        log_stage("before allocate image/output/debug bos");
        auto image_bo =
            xrt::bo(device, srad_cfg::kBoardImageBytes, toppl[0].group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, toppl[0].group_id(1));
        auto debug_bo = xrt::bo(device, 4096, q0ctrl.group_id(0));
        log_stage("after allocate image/output/debug bos");

        log_stage("before map image/output/debug bos");
        auto image_map = image_bo.map<float*>();
        auto output_map = output_bo.map<float*>();
        auto debug_map = debug_bo.map<float*>();
        log_stage("after map image/output/debug bos");

        log_stage("before copy input to image map");
        std::memcpy(image_map, input.data(), srad_cfg::kBoardImageBytes);
        std::memset(debug_map, 0, 4096);
        log_stage("after copy input to image map");

        PipelineTiming timing;
        const auto total_t0 = Clock::now();
        auto stage_t0 = Clock::now();
        log_stage("before image bo sync TO_DEVICE");
        image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        log_stage("after image bo sync TO_DEVICE");
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());
        debug_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto pl_t0 = Clock::now();
        stage_t0 = Clock::now();
        log_stage("before graphOursPLQ0.init");
        graphOursPLQ0.init();
        log_stage("after graphOursPLQ0.init");
        timing.graph_init_us = elapsed_us(stage_t0, Clock::now());
        graph_initialized = true;

        stage_t0 = Clock::now();
        log_stage("before graphOursPLQ0.run");
        graphOursPLQ0.run(graph_rows);
        log_stage("after graphOursPLQ0.run");
        timing.graph_run_submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        log_stage("before q0ctrl launch");
        auto q0_run = q0ctrl(debug_bo, active_iters);
        log_stage("after q0ctrl launch");
        timing.q0_launch_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        log_stage("before toppl launch");
        std::vector<xrt::run> toppl_runs;
        toppl_runs.reserve(srad_cfg::kTopPlWorkers);
        for (int worker = 0; worker < srad_cfg::kTopPlWorkers; ++worker) {
            toppl_runs.push_back(
                toppl[worker](image_bo, output_bo, active_iters, worker));
        }
        log_stage("after toppl launch");
        timing.toppl_launch_us = elapsed_us(stage_t0, Clock::now());
        timing.submit_us = timing.graph_init_us +
                           timing.graph_run_submit_us +
                           timing.q0_launch_us +
                           timing.toppl_launch_us;

        stage_t0 = Clock::now();
        log_stage("before toppl runs wait");
        for (int worker = 0; worker < srad_cfg::kTopPlWorkers; ++worker) {
            std::fprintf(stderr, "[diag] before TopPL_%d wait\n", worker);
            std::fflush(stderr);
            toppl_runs[worker].wait();
            std::fprintf(stderr, "[diag] after TopPL_%d wait\n", worker);
            std::fflush(stderr);
        }
        log_stage("after toppl runs wait");
        timing.toppl_wait_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        log_stage("before q0_run.wait");
        q0_run.wait();
        log_stage("after q0_run.wait");
        timing.q0_wait_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        log_stage("before graphOursPLQ0.wait");
        graphOursPLQ0.wait();
        log_stage("after graphOursPLQ0.wait");
        timing.graph_wait_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        log_stage("before graphOursPLQ0.end");
        graphOursPLQ0.end();
        log_stage("after graphOursPLQ0.end");
        timing.graph_end_us = elapsed_us(stage_t0, Clock::now());
        const auto pl_t1 = Clock::now();
        timing.wait_all_us = elapsed_us(pl_t0, pl_t1);

        std::printf("pl_total_us : %lld\n", timing.wait_all_us);

        graph_initialized = false;

        if (dump_output) {
            stage_t0 = Clock::now();
            log_stage("before output bo sync FROM_DEVICE");
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            log_stage("after output bo sync FROM_DEVICE");
            timing.bo_from_device_us = elapsed_us(stage_t0, Clock::now());
            log_stage("before dump_output_matrix");
            dump_output_matrix(output_path, output_map);
            log_stage("after dump_output_matrix");
        }

        const long long total_us = elapsed_us(total_t0, Clock::now());

        std::printf("timing us: h2d_us=%lld submit_us=%lld graph_init_us=%lld graph_run_submit_us=%lld q0_launch_us=%lld toppl_launch_us=%lld pl_total_us=%lld toppl_wait_us=%lld q0_wait_us=%lld graph_wait_us=%lld graph_end_us=%lld d2h_us=%lld total_us=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.graph_init_us,
                    timing.graph_run_submit_us,
                    timing.q0_launch_us,
                    timing.toppl_launch_us,
                    timing.wait_all_us,
                    timing.toppl_wait_us,
                    timing.q0_wait_us,
                    timing.graph_wait_us,
                    timing.graph_end_us,
                    timing.bo_from_device_us,
                    total_us);
        std::printf("ddr_to_ddr_kernel_us: %lld (legacy alias of pl_total_us)\n",
                    timing.wait_all_us);
        if (dump_output) {
            std::printf("[host] output       : %s\n", output_path.c_str());
        }

        log_stage("before xrtDeviceClose");
        xrtDeviceClose(dhdl);
        log_stage("after xrtDeviceClose");
        dhdl = nullptr;
    } catch (const std::exception& ex) {
        if (graph_initialized) {
            try {
                log_stage("exception cleanup: before graphOursPLQ0.end");
                graphOursPLQ0.end();
                log_stage("exception cleanup: after graphOursPLQ0.end");
            } catch (...) {
            }
        }
        if (dhdl) {
            log_stage("exception cleanup: before xrtDeviceClose");
            xrtDeviceClose(dhdl);
            log_stage("exception cleanup: after xrtDeviceClose");
        }
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
