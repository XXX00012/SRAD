// Spill SRAD mapping:
//   PL TopPL CUs stream 16x16 tiles to AIE lanes.
//   Each TopPL controls srad_cfg::kWorkerLanes AIE lanes.
//   PL Q0Ctrl computes global q0sqr from per-worker partial stats.
//   AIE K1 srad_local_q sends center/south/east coeff value-tag planes.
//   AIE K2 srad_coeff_update recomputes dN/dS/dW/dE from its J input and
//   applies one 16x16 SRAD tile update.
//
// The host never materializes d_c/dN/dS/dW/dE and never feeds host-computed
// q0sqr to AIE in the formal path. q0sqr_ref is CPU-side validation only;
// AIE receives the PL-computed q0sqr embedded in J tile padding.

#include "TopGraph.h"
#include "ProcessUnit/include.h"

#include <adf/adf_api/XRTConfig.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>

#include <algorithm>
#include <atomic>
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

constexpr int PREVIEW = 16;
constexpr int PL_WORKERS = srad_cfg::kTopPlWorkers;
constexpr int DIAG_POLL_MS = 1000;
constexpr int TOPPL_DEBUG_BASE = srad_cfg::kOutputElems;
constexpr int Q0_DEBUG_SLOTS = PL_WORKERS + 2;
constexpr int Q0_DEBUG_BASE = TOPPL_DEBUG_BASE + PL_WORKERS;
constexpr int DEBUG_FLOATS = PL_WORKERS + Q0_DEBUG_SLOTS;
constexpr int OUTPUT_BO_FLOATS = srad_cfg::kOutputElems + DEBUG_FLOATS;
constexpr int DEBUG_BYTES = DEBUG_FLOATS * srad_cfg::kScalarBytes;
constexpr int OUTPUT_BO_BYTES = OUTPUT_BO_FLOATS * srad_cfg::kScalarBytes;
constexpr float kCompareTol = 1.0e-5f;
using Clock = std::chrono::high_resolution_clock;

struct PipelineTiming {
    long long bo_to_device_us = 0;
    long long submit_us = 0;
    long long wait_all_us = 0;
    long long toppl_wait_us = 0;
    long long q0_wait_us = 0;
    long long graph_wait_us = 0;
    long long bo_from_device_us = 0;
};

struct CommonTiming {
    long long data_transfer_us = 0;
    long long logic_compute_us = 0;
    long long end_to_end_us = 0;
};

struct ReferenceData {
    float q0sqr = 0.0f;
    std::vector<float> j_next;
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
    std::printf("TopPL timing us: h2d=%lld pl_stream_compute_store=%lld d2h=%lld total=%lld\n",
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

void dump_float_file(const std::string& path, const std::vector<float>& buf) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s for write\n", path.c_str());
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

void print_preview(const char* tag, const std::vector<float>& buf) {
    std::printf("%s", tag);
    const int n = std::min<int>(PREVIEW, buf.size());
    for (int i = 0; i < n; ++i) {
        std::printf(" %.9g", buf[i]);
    }
    std::printf("\n");
}

float compute_q0sqr_reference(const std::vector<float>& image) {
    float sum = 0.0f;
    float sum2 = 0.0f;

    for (float v : image) {
        sum += v;
        sum2 += v * v;
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

    const float q0_den = q0sqr * (srad_math::kOne + q0sqr);
    if (jc == 0.0f || q0_den == 0.0f) {
        return 1.0f;
    }

    const float g2 =
        (dN * dN + dS * dS + dW * dW + dE * dE) / (jc * jc);
    const float lap = (dN + dS + dW + dE) / jc;
    const float num =
        srad_math::kHalf * g2 -
        srad_math::kOneSixteenth * lap * lap;
    const float den = srad_math::kOne + srad_math::kQuarter * lap;
    if (den == srad_math::kZero) {
        return srad_math::kZero;
    }
    const float qsqr = num / (den * den);
    const float ratio = (qsqr - q0sqr) / q0_den;
    const float coeff_den = srad_math::kOne + ratio;
    if (coeff_den == srad_math::kZero) {
        return srad_math::kZero;
    }
    const float c = srad_math::kOne / coeff_den;

    return srad_math::clamp01(c);
}

float sample_zero(const std::vector<float>& image, int r, int c) {
    if (r < 0 || r >= srad_cfg::kRows ||
        c < 0 || c >= srad_cfg::kCols) {
        return 0.0f;
    }
    return image[srad_math::image_index(r, c)];
}

float compute_c_zero_oob(const std::vector<float>& image,
                         int r,
                         int c,
                         float q0sqr) {
    const float jc = sample_zero(image, r, c);
    const float dN = sample_zero(image, r - 1, c) - jc;
    const float dS = sample_zero(image, r + 1, c) - jc;
    const float dW = sample_zero(image, r, c - 1) - jc;
    const float dE = sample_zero(image, r, c + 1) - jc;
    return compute_c_reference(jc, dN, dS, dW, dE, q0sqr);
}

ReferenceData cpu_reference(const std::vector<float>& image, float lambda) {
    ReferenceData ref;
    ref.q0sqr = compute_q0sqr_reference(image);
    ref.j_next.assign(srad_cfg::kOutputElems, 0.0f);

    for (int i = srad_cfg::kOutputFirstRow;
         i <= srad_cfg::kOutputLastRow;
         ++i) {
        for (int j = srad_cfg::kOutputFirstCol;
             j <= srad_cfg::kOutputLastCol;
             ++j) {
            const float JC = sample_zero(image, i, j);
            const float dN = sample_zero(image, i - 1, j) - JC;
            const float dS = sample_zero(image, i + 1, j) - JC;
            const float dW = sample_zero(image, i, j - 1) - JC;
            const float dE = sample_zero(image, i, j + 1) - JC;
            const float coeff = compute_c_zero_oob(image, i, j, ref.q0sqr);
            const float coeff_south =
                compute_c_zero_oob(image, i + 1, j, ref.q0sqr);
            const float coeff_east =
                compute_c_zero_oob(image, i, j + 1, ref.q0sqr);

            const float D =
                coeff * dN +
                coeff_south * dS +
                coeff * dW +
                coeff_east * dE;

            const int out_r = i - srad_cfg::kOutputFirstRow;
            const int out_c = j - srad_cfg::kOutputFirstCol;
            ref.j_next[out_r * srad_cfg::kOutputCols + out_c] =
                JC + srad_math::kQuarter * lambda * D;
        }
    }

    return ref;
}

ReferenceData cpu_reference_iterations(const std::vector<float>& image,
                                       float lambda,
                                       int iter_cnt) {
    std::vector<float> cur = image;
    ReferenceData ref;

    for (int iter = 0; iter < iter_cnt; ++iter) {
        ref = cpu_reference(cur, lambda);
        cur = ref.j_next;
    }

    return ref;
}

bool is_compare_pixel(int idx) {
    (void)idx;
    return true;
}

int toppl_debug_index(int worker_id) {
    return TOPPL_DEBUG_BASE + worker_id;
}

void print_debug_status(const float* output_map) {
    std::printf(" dbg_toppl=[");
    for (int i = 0; i < PL_WORKERS; ++i) {
        const int idx = toppl_debug_index(i);
        std::printf("%s%d:%.0f@%d",
                    (i == 0) ? "" : " ",
                    i,
                    output_map[idx],
                    idx);
    }
    std::printf("] q0dbg=[");
    for (int i = 0; i < Q0_DEBUG_SLOTS; ++i) {
        std::printf("%s%.0f",
                    (i == 0) ? "" : " ",
                    output_map[Q0_DEBUG_BASE + i]);
    }
    std::printf("]");
}

void compare_output(const std::vector<float>& got,
                    const std::vector<float>& gold) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_idx = -1;
    int mismatch_count = 0;
    int compared_count = 0;

    for (int i = 0; i < srad_cfg::kOutputElems; ++i) {
        if (!is_compare_pixel(i)) continue;
        const float abs_err = std::fabs(got[i] - gold[i]);
        const float denom = std::max(std::fabs(gold[i]), 1.0e-12f);
        const float rel_err = abs_err / denom;
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_idx = i;
        }
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > kCompareTol) ++mismatch_count;
        ++compared_count;
    }

    std::printf("compared_pixels      : %d\n", compared_count);
    std::printf("max_abs_error_float : %.9g\n", max_abs);
    std::printf("max_relative_error  : %.9g\n", max_rel);
    if (max_idx >= 0) {
        const int out_r = max_idx / srad_cfg::kOutputCols;
        const int out_c = max_idx % srad_cfg::kOutputCols;
        std::printf("max_error_location  : compact output row %d col %d, source (%d,%d)\n",
                    out_r,
                    out_c,
                    out_r + srad_cfg::kOutputFirstRow,
                    out_c + srad_cfg::kOutputFirstCol);
    }
    std::printf("mismatch_count_tol_%g: %d\n", kCompareTol, mismatch_count);
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
                     "Usage: %s <xclbin> [iter_cnt] [input_txt] [output_txt]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string xclbin_path = argv[1];
    const int iter_cnt =
        (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const std::string input_path =
        (argc >= 4) ? argv[3] : "./data/plio_ours_j.txt";
    const std::string output_path =
        (argc >= 5) ? argv[4] : "./data/aie_j_next.txt";
    const float lambda = srad_cfg::kLambdaDefault;

    if (iter_cnt < 1 || iter_cnt > srad_cfg::kSradIterations) {
        std::fprintf(stderr,
                     "[error] Ours supports 1..%d iteration(s) in this build; got %d\n",
                     srad_cfg::kSradIterations,
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels);
    if (!load_float_file(input_path, image)) {
        std::fprintf(stderr, "[warn] fallback to deterministic input\n");
        fill_default_image(image);
    }
    const ReferenceData ref = cpu_reference_iterations(image, lambda, iter_cnt);

    std::printf("mapping               : %d x TopPL + Q0Ctrl + %d-lane AIE K1/K2 (%d lane(s)/TopPL)\n",
                srad_cfg::kTopPlWorkers,
                srad_cfg::kParallelLanes,
                srad_cfg::kWorkerLanes);
    std::printf("trace kernels         : srad_local_q, srad_coeff_update\n");
    std::printf("iterations            : %d\n", iter_cnt);
    std::printf("image size            : %dx%d (%d pixels)\n",
                srad_cfg::kRows, srad_cfg::kCols, srad_cfg::kPixels);
    std::printf("tile schedule         : %dx%d output tiles, output %dx%d, input logical %dx%d, input physical row=%d; %d graph firing(s)/lane, %d lanes, %d input floats/firing/lane, %d output floats/firing/lane\n",
                srad_cfg::kTileRowCount,
                srad_cfg::kTileColCount,
                srad_cfg::kOutputTileRows,
                srad_cfg::kOutputTileCols,
                srad_cfg::kInputLogicalRows,
                srad_cfg::kInputLogicalCols,
                srad_cfg::kInputRowElems,
                srad_cfg::kGraphRunIterations,
                srad_cfg::kParallelLanes,
                srad_cfg::kImageInputSampleElems,
                srad_cfg::kOutputSampleElems);
    std::printf("compact output        : %dx%d, source rows %d..%d cols %d..%d\n",
                srad_cfg::kOutputRows,
                srad_cfg::kOutputCols,
                srad_cfg::kOutputFirstRow,
                srad_cfg::kOutputLastRow,
                srad_cfg::kOutputFirstCol,
                srad_cfg::kOutputLastCol);
    std::printf("data size bytes       : J=%d J_next=%d coeff_mid=%d grad_mid=0(recomputed in K2)\n",
                srad_cfg::kImageBytes,
                srad_cfg::kOutputBytes,
                srad_cfg::kMidBytes);
    std::printf("output BO bytes       : result=%d debug_tail=%d total=%d\n",
                srad_cfg::kOutputBytes,
                DEBUG_BYTES,
                OUTPUT_BO_BYTES);
    std::printf("float lanes per packet: %d\n", srad_cfg::kLanes);
    std::printf("lambda float32        : %.9g\n", lambda);
    std::printf("q0sqr_ref last iter   : %.9g\n", ref.q0sqr);
    print_preview("input preview:", image);

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
        const auto t0 = Clock::now();

        auto stage_t0 = Clock::now();
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        std::printf("---- %d x TopPL + Q0Ctrl + GraphOursPLQ0, %d AIE lanes ----\n",
                    srad_cfg::kTopPlWorkers,
                    srad_cfg::kParallelLanes);
        std::printf("PL input              : DDR J float[%d]\n",
                    srad_cfg::kPixels);
        std::printf("PL q0/output packing  : each TopPL merges %d lane stats; Q0Ctrl reduces %d worker partial stats; q0sqr embedded in first padding column of each 19x24 J tile\n",
                    srad_cfg::kWorkerLanes,
                    srad_cfg::kTopPlWorkers);
        std::printf("AIE q0 input          : K1 reads q0sqr from the J tile padding slot\n");
        std::printf("AIE tile firing       : graph.run(%d), %d lanes, %d input floats per firing/lane, %d output floats per firing/lane\n",
                    srad_cfg::kGraphRunIterations * iter_cnt,
                    srad_cfg::kParallelLanes,
                    srad_cfg::kImageInputSampleElems,
                    srad_cfg::kOutputSampleElems);
        std::printf("PL output store       : one 16x16 result tile plus tile_sum/tile_sum2 per firing; final compact %dx%d output\n",
                    srad_cfg::kOutputRows,
                    srad_cfg::kOutputCols);
        std::printf("AIE tile payload      : input %dx%d halo-padded floats (%d), output %dx%d floats (%d)\n",
                    srad_cfg::kInputLogicalRows,
                    srad_cfg::kInputRowElems,
                    srad_cfg::kImageInputSampleElems,
                    srad_cfg::kOutputTileRows,
                    srad_cfg::kOutputTileCols,
                    srad_cfg::kOutputSampleElems);
        std::printf("AIE K1                : srad_local_q sends center/south/east coeff value-tag planes\n");
        std::printf("AIE K2                : srad_coeff_update decodes coeff value-tags, recomputes gradients from J, and applies one 16x16 tile update\n");
        std::fflush(stdout);

        std::printf("[stage] graphOursPLQ0.init begin\n");
        std::fflush(stdout);
        graphOursPLQ0.init();
        std::printf("[stage] graphOursPLQ0.init done\n");
        std::fflush(stdout);

        stage_t0 = Clock::now();
        std::printf("[stage] Q0Ctrl_1 submit begin\n");
        std::fflush(stdout);
        auto q0_run = q0_kernel(output_bo, iter_cnt);
        std::printf("[stage] Q0Ctrl_1 submit done\n");
        std::fflush(stdout);

        std::vector<xrt::run> toppl_runs;
        toppl_runs.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            std::printf("[stage] TopPL_%d submit begin\n", i);
            std::fflush(stdout);
            toppl_runs.emplace_back(toppl_kernels[i](input_bo,
                                                     output_bo,
                                                     iter_cnt,
                                                     i));
            std::printf("[stage] TopPL_%d submit done\n", i);
            std::fflush(stdout);
        }

        std::printf("[stage] graphOursPLQ0.run begin\n");
        std::fflush(stdout);
        graphOursPLQ0.run(srad_cfg::kGraphRunIterations * iter_cnt);
        std::printf("[stage] graphOursPLQ0.run done\n");
        std::fflush(stdout);
        timing.submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        std::printf("[stage] wait begin\n");
        std::fflush(stdout);

        std::atomic_bool toppl_done[PL_WORKERS];
        std::atomic_bool toppl_failed[PL_WORKERS];
        long long toppl_wait_elapsed[PL_WORKERS] = {};
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_done[i].store(false, std::memory_order_relaxed);
            toppl_failed[i].store(false, std::memory_order_relaxed);
        }
        std::atomic_bool q0_done(false);
        std::atomic_bool q0_failed(false);
        std::atomic_bool graph_done(false);
        std::atomic_bool graph_failed(false);
        long long q0_wait_elapsed = 0;
        long long graph_wait_elapsed = 0;

        std::vector<std::thread> toppl_wait_threads;
        toppl_wait_threads.reserve(PL_WORKERS);
        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_wait_threads.emplace_back([&, i]() {
                const auto wait_t0 = Clock::now();
                std::printf("[stage] TopPL_%d wait begin\n", i);
                std::fflush(stdout);
                try {
                    toppl_runs[i].wait();
                    toppl_wait_elapsed[i] = elapsed_us(wait_t0, Clock::now());
                    toppl_done[i].store(true, std::memory_order_relaxed);
                    std::printf("[stage] TopPL_%d wait done (%lld us)\n",
                                i,
                                toppl_wait_elapsed[i]);
                    std::fflush(stdout);
                } catch (const std::exception& ex) {
                    toppl_wait_elapsed[i] = elapsed_us(wait_t0, Clock::now());
                    toppl_failed[i].store(true, std::memory_order_relaxed);
                    toppl_done[i].store(true, std::memory_order_relaxed);
                    std::fprintf(stderr,
                                 "[error] TopPL_%d wait failed: %s\n",
                                 i,
                                 ex.what());
                    std::fflush(stderr);
                }
            });
        }

        std::thread q0_wait_thread([&]() {
            const auto wait_t0 = Clock::now();
            std::printf("[stage] Q0Ctrl_1 wait begin\n");
            std::fflush(stdout);
            try {
                q0_run.wait();
                q0_wait_elapsed = elapsed_us(wait_t0, Clock::now());
                q0_done.store(true, std::memory_order_relaxed);
                std::printf("[stage] Q0Ctrl_1 wait done (%lld us)\n",
                            q0_wait_elapsed);
                std::fflush(stdout);
            } catch (const std::exception& ex) {
                q0_wait_elapsed = elapsed_us(wait_t0, Clock::now());
                q0_failed.store(true, std::memory_order_relaxed);
                q0_done.store(true, std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[error] Q0Ctrl_1 wait failed: %s\n",
                             ex.what());
                std::fflush(stderr);
            }
        });

        std::thread graph_wait_thread;
        graph_wait_thread = std::thread([&]() {
            const auto wait_t0 = Clock::now();
            std::printf("[stage] graphOursPLQ0.wait begin\n");
            std::fflush(stdout);
            try {
                graphOursPLQ0.wait();
                graph_wait_elapsed = elapsed_us(wait_t0, Clock::now());
                graph_done.store(true, std::memory_order_relaxed);
                std::printf("[stage] graphOursPLQ0.wait done (%lld us)\n",
                            graph_wait_elapsed);
                std::fflush(stdout);
            } catch (const std::exception& ex) {
                graph_wait_elapsed = elapsed_us(wait_t0, Clock::now());
                graph_failed.store(true, std::memory_order_relaxed);
                graph_done.store(true, std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[error] graphOursPLQ0.wait failed: %s\n",
                             ex.what());
                std::fflush(stderr);
            }
        });

        auto all_toppl_done = [&]() {
            for (int i = 0; i < PL_WORKERS; ++i) {
                if (!toppl_done[i].load(std::memory_order_relaxed)) {
                    return false;
                }
            }
            return true;
        };

        while (!all_toppl_done() ||
               !q0_done.load(std::memory_order_relaxed) ||
               !graph_done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(DIAG_POLL_MS));
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::printf("[diag] wait_status elapsed_ms=%lld graph=%s%s q0=%s%s toppl=",
                        elapsed_us(stage_t0, Clock::now()) / 1000,
                        graph_done.load(std::memory_order_relaxed) ? "done" : "pending",
                        graph_failed.load(std::memory_order_relaxed) ? "(failed)" : "",
                        q0_done.load(std::memory_order_relaxed) ? "done" : "pending",
                        q0_failed.load(std::memory_order_relaxed) ? "(failed)" : "");
            for (int i = 0; i < PL_WORKERS; ++i) {
                std::printf("%s%d:%s%s",
                            (i == 0) ? "[" : " ",
                            i,
                            toppl_done[i].load(std::memory_order_relaxed) ? "done" : "pending",
                            toppl_failed[i].load(std::memory_order_relaxed) ? "(failed)" : "");
            }
            std::printf("]");
            print_debug_status(output_map);
            std::printf("\n");
            std::fflush(stdout);
        }

        for (int i = 0; i < PL_WORKERS; ++i) {
            toppl_wait_threads[i].join();
            timing.toppl_wait_us =
                std::max(timing.toppl_wait_us, toppl_wait_elapsed[i]);
        }
        q0_wait_thread.join();
        if (graph_wait_thread.joinable()) {
            graph_wait_thread.join();
        }
        timing.q0_wait_us = q0_wait_elapsed;
        timing.graph_wait_us = graph_wait_elapsed;
        timing.wait_all_us = elapsed_us(stage_t0, Clock::now());

        bool any_wait_failed =
            q0_failed.load(std::memory_order_relaxed) ||
            graph_failed.load(std::memory_order_relaxed);
        for (int i = 0; i < PL_WORKERS; ++i) {
            any_wait_failed =
                any_wait_failed ||
                toppl_failed[i].load(std::memory_order_relaxed);
        }
        if (any_wait_failed) {
            throw std::runtime_error("TopPL, Q0Ctrl, or graph wait failed");
        }

        std::printf("[stage] graphOursPLQ0.end begin\n");
        std::fflush(stdout);
        graphOursPLQ0.end();
        std::printf("[stage] graphOursPLQ0.end done\n");
        std::fflush(stdout);


        const auto t1 = Clock::now();
        const long long kernel_phase_us = elapsed_us(t0, t1);

        stage_t0 = Clock::now();
        const bool final_in_output = ((iter_cnt & 1) != 0);
        if (final_in_output) {
            output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        } else {
            input_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }
        timing.bo_from_device_us = elapsed_us(stage_t0, Clock::now());
        const long long total_us = elapsed_us(t0, Clock::now());
        const float* result_map = final_in_output ? output_map : input_map;
        std::vector<float> got(result_map, result_map + srad_cfg::kOutputElems);
        print_preview("output preview:", got);
        compare_output(got, ref.j_next);

        print_toppl_timing(timing);
        std::printf("wait timing us: all=%lld toppl_max=%lld q0=%lld graph=%lld\n",
                    timing.wait_all_us,
                    timing.toppl_wait_us,
                    timing.q0_wait_us,
                    timing.graph_wait_us);
        std::printf("staged graph/PL time before d2h: %lld us\n",
                    kernel_phase_us);

        std::printf("timing us: h2d=%lld submit=%lld wait_all=%lld toppl_max=%lld q0_wait=%lld graph_wait=%lld d2h=%lld total=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.wait_all_us,
                    timing.toppl_wait_us,
                    timing.q0_wait_us,
                    timing.graph_wait_us,
                    timing.bo_from_device_us,
                    total_us);

        CommonTiming common;
        common.data_transfer_us =
            timing.bo_to_device_us + timing.bo_from_device_us;
        common.logic_compute_us = timing.wait_all_us;
        common.end_to_end_us = total_us;
        print_common_timing(common);

        dump_float_file(output_path, got);

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
