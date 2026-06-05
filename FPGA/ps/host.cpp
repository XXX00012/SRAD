// FPGA-v5-style SRAD AIE baseline:
//   DDR -> PL transport shim -> one fused AIE kernel -> PL transport shim -> DDR
//
// All PLIO-visible data is float32, matching Rodinia OpenCL SRAD's `fp float`.
// Phase 1 computes q0sqr inside the AIE kernel. Phase 2 rereads J and fuses
// coefficient generation with the update.

#include "TopGraph.h"
#include "ProcessUnit/include.h"

#include <adf/adf_api/XRTConfig.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

GraphFpgaV5 graphFpgaV5("fpga_v5");

namespace {

constexpr int PREVIEW = 16;
constexpr float kCompareTol = 1.0e-5f;
using Clock = std::chrono::high_resolution_clock;

struct PipelineTiming {
    long long submit_us = 0;
    long long wait_inputs_us = 0;
    long long wait_output_us = 0;
    long long wait_graph_us = 0;
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

void print_wall_timing(const char* name, const PipelineTiming& timing) {
    const long long total =
        timing.submit_us + timing.wait_inputs_us +
        timing.wait_output_us + timing.wait_graph_us;
    std::printf("%s hw_emu wall timing us: submit=%lld wait_inputs=%lld wait_output=%lld wait_graph=%lld total=%lld\n",
                name,
                timing.submit_us,
                timing.wait_inputs_us,
                timing.wait_output_us,
                timing.wait_graph_us,
                total);
}

void print_common_timing(const char* name, const CommonTiming& timing) {
    std::printf("---- Common timing summary (host wall us): %s ----\n", name);
    std::printf("data_transfer_us: %lld\n", timing.data_transfer_us);
    std::printf("logic_compute_us: %lld\n", timing.logic_compute_us);
    std::printf("end_to_end_us   : %lld\n", timing.end_to_end_us);
}

bool load_float_file(const std::string& path, std::vector<float>& buf) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::fprintf(stderr, "[warn] cannot open %s\n", path.c_str());
        return false;
    }

    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            float v = 0.0f;
            if (!(fin >> v)) {
                std::fprintf(stderr,
                             "[warn] %s element count mismatch, expect %d values\n",
                             path.c_str(), srad_cfg::kPixels);
                return false;
            }
            buf[srad_math::image_index(r, c)] = v;
        }
    }
    return true;
}

void fill_default_image(std::vector<float>& image) {
    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            image[srad_math::image_index(r, c)] =
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

    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            if (c) fout << ' ';
            fout << buf[srad_math::image_index(r, c)];
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
    return variance / (mean * mean);
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
    const float l = (dN + dS + dW + dE) / jc;
    const float num = 0.5f * g2 - (1.0f / 16.0f) * l * l;
    float den = 1.0f + 0.25f * l;
    const float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    const float c = 1.0f / (1.0f + den);

    return srad_math::clamp01(c);
}

ReferenceData cpu_reference(const std::vector<float>& image, float lambda) {
    ReferenceData ref;
    ref.q0sqr = compute_q0sqr_reference(image);
    ref.j_next.assign(srad_cfg::kPixels, 0.0f);

    std::vector<float> c_plane(srad_cfg::kPixels, 0.0f);

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            const int idx = srad_math::image_index(i, j);
            const int iN = srad_math::north_row(i);
            const int iS = srad_math::south_row(i);
            const int jW = srad_math::west_col(j);
            const int jE = srad_math::east_col(j);

            const float JC = image[idx];
            const float dN = image[srad_math::image_index(iN, j)] - JC;
            const float dS = image[srad_math::image_index(iS, j)] - JC;
            const float dW = image[srad_math::image_index(i, jW)] - JC;
            const float dE = image[srad_math::image_index(i, jE)] - JC;

            c_plane[idx] = compute_c_reference(JC, dN, dS, dW, dE, ref.q0sqr);
        }
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            const int idx = srad_math::image_index(i, j);
            const int iS = srad_math::south_row(i);
            const int jE = srad_math::east_col(j);
            const int south_idx = srad_math::image_index(iS, j);
            const int east_idx = srad_math::image_index(i, jE);

            const float JC = image[idx];
            const float dN = image[srad_math::image_index(srad_math::north_row(i), j)] - JC;
            const float dS = image[south_idx] - JC;
            const float dW = image[srad_math::image_index(i, srad_math::west_col(j))] - JC;
            const float dE = image[east_idx] - JC;

            const float D =
                c_plane[idx] * dN +
                c_plane[south_idx] * dS +
                c_plane[idx] * dW +
                c_plane[east_idx] * dE;
            ref.j_next[idx] = image[idx] + 0.25f * lambda * D;
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
        if (abs_err > kCompareTol) ++mismatch_count;
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
    const int iter_cnt = (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const std::string input_path = (argc >= 4) ? argv[3] : "./data/input_32x32.txt";
    const std::string output_path = (argc >= 5) ? argv[4] : "./data/aie_j_next.txt";
    const float lambda = (argc >= 6) ? static_cast<float>(std::atof(argv[5]))
                                     : srad_cfg::kLambdaDefault;

    if (iter_cnt != 1) {
        std::fprintf(stderr,
                     "[error] this FPGA-v5-style baseline supports one iteration; got %d\n",
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels);
    if (!load_float_file(input_path, image)) {
        std::fprintf(stderr, "[warn] fallback to deterministic input\n");
        fill_default_image(image);
    }

    const ReferenceData ref = cpu_reference(image, lambda);

    std::printf("image size            : %dx%d (%d pixels)\n", ROW, COL, srad_cfg::kPixels);
    std::printf("float lanes per packet: %d\n", srad_cfg::kLanes);
    std::printf("q0sqr_ref float32     : %.9g\n", ref.q0sqr);
    std::printf("lambda float32        : %.9g\n", lambda);
    std::printf("q0sqr_aie             : internal to fused kernel, no baseline output port\n");
    print_preview("input preview:", image);

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

    xuid_t uuid;
    xrtDeviceGetXclbinUUID(dhdl, uuid);
    adf::registerXRT(dhdl, uuid);

    auto load_fpga = open_kernel_or_die(device, xrt_uuid, "LoadFpgaV5");
    auto store_fpga = open_kernel_or_die(device, xrt_uuid, "StoreFpgaV5");

    auto image_bo = xrt::bo(device, srad_cfg::kImageBytes, load_fpga.group_id(0));
    auto out_bo = xrt::bo(device, srad_cfg::kImageBytes, store_fpga.group_id(0));

    auto image_map = image_bo.map<float*>();
    auto out_map = out_bo.map<float*>();

    std::memcpy(image_map, image.data(), srad_cfg::kImageBytes);
    std::memset(out_map, 0, srad_cfg::kImageBytes);
    image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    PipelineTiming timing;
    const auto t0 = Clock::now();

    std::printf("---- LoadFpgaV5 + GraphFpgaV5 + StoreFpgaV5: DDR -> PL -> AIE -> PL -> DDR ----\n");
    graphFpgaV5.init();

    auto stage_t0 = Clock::now();
    auto sink_run = store_fpga(out_bo);
    graphFpgaV5.run(1);
    auto source_run = load_fpga(image_bo, lambda);
    source_run.wait();
    timing.wait_inputs_us = elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    graphFpgaV5.wait();
    timing.wait_graph_us = elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    sink_run.wait();
    timing.wait_output_us = elapsed_us(stage_t0, Clock::now());

    graphFpgaV5.end();

    const auto t1 = Clock::now();
    const auto dur_us = elapsed_us(t0, t1);

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<float> got(out_map, out_map + srad_cfg::kPixels);
    print_preview("output preview:", got);
    compare_output(got, ref.j_next);
    print_wall_timing("GraphFpgaV5", timing);
    std::printf("fused graph hw_emu wall time: %lld us\n",
                static_cast<long long>(dur_us));

    CommonTiming common;
    common.data_transfer_us =
        timing.wait_inputs_us + timing.wait_output_us;
    common.logic_compute_us = timing.wait_graph_us;
    common.end_to_end_us = dur_us;
    print_common_timing("FPGA", common);

    dump_float_file(output_path, got);

    (void)dhdl;

    return EXIT_SUCCESS;
}
