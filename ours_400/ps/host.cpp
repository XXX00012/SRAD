// Ours SRAD mapping:
//   4 PL TopPLWorker CUs stream 200 16x16 tiles to 200 AIE lanes.
//   PL Q0Ctrl computes global q0sqr from 4 partial sum/sum2 streams.
//   AIE K1 srad_local_q sends center/south/east coeff value-tag planes.
//   AIE K2 srad_coeff_update recomputes dN/dS/dW/dE from J and applies one
//   16x16 SRAD tile update.
//
// The host never materializes d_c/dN/dS/dW/dE and never feeds host-computed
// q0sqr to AIE in the formal path. AIE receives the PL-computed q0sqr
// embedded in J tile padding.

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

GraphOursPLQ0 graphOursPLQ0("ours_plq0");

namespace {

constexpr int PREVIEW = 16;
constexpr int PL_WORKERS = srad_cfg::kTopPlWorkers;
using Clock = std::chrono::high_resolution_clock;

struct PipelineTiming {
    long long bo_to_device_us = 0;
    long long submit_us = 0;
    long long toppl_wait_us = 0;
    long long graph_wait_us = 0;
    long long bo_from_device_us = 0;
};

struct CommonTiming {
    long long data_transfer_us = 0;
    long long logic_compute_us = 0;
    long long end_to_end_us = 0;
};

long long elapsed_us(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
}

void print_common_timing(const CommonTiming& timing) {
    std::printf("---- Common timing summary (host wall us): Ours ----\n");
    std::printf("data_transfer_us: %lld\n", timing.data_transfer_us);
    std::printf("logic_compute_us: %lld\n", timing.logic_compute_us);
    std::printf("end_to_end_us   : %lld\n", timing.end_to_end_us);
}

void print_stage_timing(const char* name,
                        long long in_us,
                        long long compute_us,
                        long long out_us) {
    std::printf("%s timing us: plio_in=%lld compute=%lld plio_out=%lld total=%lld\n",
                name,
                in_us,
                compute_us,
                out_us,
                in_us + compute_us + out_us);
}

void print_toppl_timing(const PipelineTiming& timing) {
    const long long total =
        timing.bo_to_device_us + timing.toppl_wait_us +
        timing.bo_from_device_us;
    std::printf("PL timing us: h2d=%lld workers_q0_stream_compute_store=%lld d2h=%lld total=%lld\n",
                timing.bo_to_device_us,
                timing.toppl_wait_us,
                timing.bo_from_device_us,
                total);
}

bool load_float_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s\n", path.c_str());
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
                     "[warn] %s element count mismatch: got %d, expect %zu\n",
                     path.c_str(), cnt, buf.size());
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

void print_preview(const char* tag, const float* buf, int count) {
    std::printf("%s", tag);
    const int n = std::min(PREVIEW, count);
    for (int i = 0; i < n; ++i) {
        std::printf(" %.9g", buf[i]);
    }
    std::printf("\n");
}

void print_checksum(const char* tag, const float* buf, int count) {
    double sum = 0.0;
    double sum2 = 0.0;
    float min_val = count > 0 ? buf[0] : 0.0f;
    float max_val = min_val;

    for (int i = 0; i < count; ++i) {
        const float v = buf[i];
        sum += static_cast<double>(v);
        sum2 += static_cast<double>(v) * static_cast<double>(v);
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }

    std::printf("%s checksum: sum=%.12g sum2=%.12g min=%.9g max=%.9g count=%d\n",
                tag,
                sum,
                sum2,
                min_val,
                max_val,
                count);
}

xrt::kernel open_toppl_worker_kernel(const xrt::device& device,
                                      const xrt::uuid& uuid,
                                      int cu_index) {
    char scoped[80];
    std::snprintf(scoped,
                  sizeof(scoped),
                  "TopPLWorker:{TopPL_%d}",
                  cu_index);
    try {
        return xrt::kernel(device, uuid, scoped);
    } catch (const std::exception&) {
        char cu_only[32];
        std::snprintf(cu_only, sizeof(cu_only), "TopPL_%d", cu_index);
        try {
            return xrt::kernel(device, uuid, cu_only);
        } catch (const std::exception&) {
            if (cu_index == 0) {
                return xrt::kernel(device, uuid, "TopPLWorker");
            }
            throw;
        }
    }
}

xrt::kernel open_q0ctrl_kernel(const xrt::device& device,
                               const xrt::uuid& uuid) {
    try {
        return xrt::kernel(device, uuid, "Q0Ctrl:{Q0Ctrl_1}");
    } catch (const std::exception&) {
        return xrt::kernel(device, uuid, "Q0Ctrl");
    }
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
    const float lambda = srad_cfg::kLambdaDefault;

    if (iter_cnt < 1 || iter_cnt > srad_cfg::kSradIterations) {
        std::fprintf(stderr,
                     "[error] Ours_400 supports 1..%d iteration(s) in this build; got %d\n",
                     srad_cfg::kSradIterations,
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels);
    if (argc >= 4) {
        const std::string input_path = argv[3];
        if (!load_float_file(input_path, image)) {
            std::fprintf(stderr, "[warn] fallback to deterministic input\n");
            fill_default_image(image);
        }
    } else {
        std::printf("[info] no input_txt provided; using deterministic generated input\n");
        fill_default_image(image);
    }

    std::printf("mapping               : Ours_400 4x TopPLWorker + Q0Ctrl + 200-lane AIE K1/K2 halo-tile coeff\n");
    std::printf("trace kernels         : srad_local_q, srad_coeff_update\n");
    std::printf("iterations            : %d\n", iter_cnt);
    std::printf("parallel lanes        : %d (%d AIE kernels, %d kernels/lane)\n",
                srad_cfg::kParallelLanes,
                srad_cfg::kTotalAieCores,
                srad_cfg::kKernelsPerParallelLane);
    std::printf("PL workers            : %d workers, %d lanes/worker, %d output groups/worker\n",
                srad_cfg::kTopPlWorkers,
                srad_cfg::kWorkerLanes,
                srad_cfg::kWorkerOutputGroups);
    std::printf("output PLIO grouping  : %d groups, pktmerge<%d> lanes/group\n",
                srad_cfg::kOutputPlioGroups,
                srad_cfg::kOutputLanesPerPlio);
    std::printf("image size            : %dx%d (%d pixels)\n",
                srad_cfg::kRows, srad_cfg::kCols, srad_cfg::kPixels);
    std::printf("tile schedule         : %dx%d output tiles, output %dx%d, input logical %dx%d, input physical row=%d; %d graph firing(s)/lane, %d input floats/firing/lane, %d output floats/firing/lane\n",
                srad_cfg::kTileRowCount,
                srad_cfg::kTileColCount,
                srad_cfg::kOutputTileRows,
                srad_cfg::kOutputTileCols,
                srad_cfg::kInputLogicalRows,
                srad_cfg::kInputLogicalCols,
                srad_cfg::kInputRowElems,
                srad_cfg::kGraphRunIterations,
                srad_cfg::kImageInputSampleElems,
                srad_cfg::kOutputSampleElems);
    std::printf("AIE tile slots        : %d valid tiles over %d scheduled lane slots\n",
                srad_cfg::kTotalTileCount,
                srad_cfg::kAieOutputTiles);
    std::printf("compact output        : %dx%d, source rows %d..%d cols %d..%d\n",
                srad_cfg::kOutputRows,
                srad_cfg::kOutputCols,
                srad_cfg::kOutputFirstRow,
                srad_cfg::kOutputLastRow,
                srad_cfg::kOutputFirstCol,
                srad_cfg::kOutputLastCol);
    std::printf("data size bytes       : J=%d J_next=%d coeff_mid=%lld\n",
                srad_cfg::kImageBytes,
                srad_cfg::kOutputBytes,
                srad_cfg::kMidBytes);
    std::printf("float lanes per packet: %d\n", srad_cfg::kLanes);
    std::printf("lambda float32        : %.9g\n", lambda);
    print_preview("input preview:", image.data(), srad_cfg::kPixels);

    try {
        auto device = xrt::device(0);
        auto xrt_uuid = device.load_xclbin(xclbin_path);

        auto dhdl = xrtDeviceOpen(0);
        if (!dhdl) {
            std::fprintf(stderr, "[error] xrtDeviceOpen failed\n");
            return EXIT_FAILURE;
        }

        if (xrtDeviceLoadXclbinFile(dhdl, xclbin_path.c_str())) {
            std::fprintf(stderr, "[error] xrtDeviceLoadXclbinFile failed\n");
            xrtDeviceClose(dhdl);
            return EXIT_FAILURE;
        }

        xuid_t adf_uuid;
        xrtDeviceGetXclbinUUID(dhdl, adf_uuid);
        adf::registerXRT(dhdl, adf_uuid);

        std::vector<xrt::kernel> toppl_kernels;
        toppl_kernels.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_kernels.emplace_back(
                open_toppl_worker_kernel(device, xrt_uuid, i));
        }
        auto q0_kernel = open_q0ctrl_kernel(device, xrt_uuid);

        auto input_bo =
            xrt::bo(device, srad_cfg::kImageBytes, toppl_kernels[0].group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kOutputBytes, toppl_kernels[0].group_id(1));
        auto input_map = input_bo.map<float*>();
        auto output_map = output_bo.map<float*>();

        std::memcpy(input_map, image.data(), srad_cfg::kImageBytes);
        std::memset(output_map, 0, srad_cfg::kOutputBytes);

        PipelineTiming timing;
        const auto t0 = Clock::now();

        auto stage_t0 = Clock::now();
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        std::printf("---- 4x TopPLWorker + Q0Ctrl + GraphOursPLQ0 ----\n");
        std::printf("PL input              : DDR J float[%d]\n",
                    srad_cfg::kPixels);
        std::printf("PL q0/output packing  : Q0Ctrl reduces 4 partial stats; q0sqr embedded in first padding column of each 19x24 J tile\n");
        std::printf("AIE q0 input          : K1 reads q0sqr from the J tile padding slot\n");
        std::printf("AIE tile firing       : graph.run(%d), %d lanes, %d input floats per firing/lane, %d output floats per firing/lane\n",
                    srad_cfg::kGraphRunIterations * iter_cnt,
                    srad_cfg::kParallelLanes,
                    srad_cfg::kImageInputSampleElems,
                    srad_cfg::kOutputSampleElems);
        std::printf("PL output store       : 4 workers read 100 grouped pktmerge<2> streams; one 16x16 result tile plus tile_sum/tile_sum2 per lane slot into final compact %dx%d output\n",
                    srad_cfg::kOutputRows,
                    srad_cfg::kOutputCols);
        std::printf("AIE tile payload      : input %dx%d halo-padded floats (%d), output %dx%d data floats + %d stats floats (%d)\n",
                    srad_cfg::kInputLogicalRows,
                    srad_cfg::kInputRowElems,
                    srad_cfg::kImageInputSampleElems,
                    srad_cfg::kOutputTileRows,
                    srad_cfg::kOutputTileCols,
                    srad_cfg::kOutputStatElems,
                    srad_cfg::kOutputSampleElems);
        std::printf("AIE K1                : srad_local_q sends center/south/east coeff value-tag planes\n");
        std::printf("AIE K2                : srad_coeff_update decodes coeff value-tags, recomputes dN/dS/dW/dE, and applies one 16x16 tile update\n");
        std::fflush(stdout);

        graphOursPLQ0.init();

        stage_t0 = Clock::now();
        auto q0_run = q0_kernel(iter_cnt);
        graphOursPLQ0.run(srad_cfg::kGraphRunIterations * iter_cnt);
        std::vector<xrt::run> toppl_runs;
        toppl_runs.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_runs.emplace_back(toppl_kernels[i](input_bo,
                                                     output_bo,
                                                     iter_cnt,
                                                     i));
        }
        timing.submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_runs[i].wait();
        }
        q0_run.wait();
        timing.toppl_wait_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        graphOursPLQ0.wait();
        timing.graph_wait_us = elapsed_us(stage_t0, Clock::now());

        graphOursPLQ0.end();

        const auto t1 = Clock::now();
        const long long kernel_phase_us = elapsed_us(t0, t1);

        stage_t0 = Clock::now();
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        timing.bo_from_device_us = elapsed_us(stage_t0, Clock::now());
        const long long total_us = elapsed_us(t0, Clock::now());
        print_preview("output preview:", output_map, srad_cfg::kOutputElems);
        print_checksum("output", output_map, srad_cfg::kOutputElems);

        print_toppl_timing(timing);
        print_stage_timing("GraphOurs",
                           timing.toppl_wait_us,
                           timing.graph_wait_us,
                           timing.bo_from_device_us);
        std::printf("staged graph/PL time before d2h: %lld us\n",
                    kernel_phase_us);

        std::printf("timing us: h2d=%lld submit=%lld toppl_plio=%lld graph_wait=%lld d2h=%lld total=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.toppl_wait_us,
                    timing.graph_wait_us,
                    timing.bo_from_device_us,
                    total_us);

        CommonTiming common;
        common.data_transfer_us =
            timing.bo_to_device_us + timing.bo_from_device_us;
        common.logic_compute_us =
            timing.toppl_wait_us + timing.graph_wait_us;
        common.end_to_end_us = total_us;
        print_common_timing(common);

        xrtDeviceClose(dhdl);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
