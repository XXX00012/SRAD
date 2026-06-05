// OpenCL-v0-faithful staged AIE reproduction:
//   prepare_kernel -> reduce_kernel -> host q0 scalar -> srad_kernel -> srad2_kernel
//
// All global arrays use OpenCL's linear layout `row + Nr * col`. PL movers
// only transfer, tile, halo-pack, and store data for PLIO; the SRAD arithmetic
// stays in AIE kernels.

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

GraphPrepare graphPrepare("prepare");
GraphReduce graphReduce("reduce");
GraphCoeff graphCoeff("coeff");
GraphUpdate graphUpdate("update");

namespace {

constexpr int PREVIEW = 16;
constexpr float kCompareTol = 1.0e-5f;
using Clock = std::chrono::steady_clock;

struct PipelineTiming {
    long long prepare_in_us = 0;
    long long prepare_compute_us = 0;
    long long prepare_out_us = 0;
    long long reduce_in_us = 0;
    long long reduce_compute_us = 0;
    long long reduce_out_us = 0;
    long long coeff_in_us = 0;
    long long coeff_compute_us = 0;
    long long coeff_out_us = 0;
    long long update_in_us = 0;
    long long update_compute_us = 0;
    long long update_out_us = 0;
};

struct CommonTiming {
    long long data_transfer_us = 0;
    long long logic_compute_us = 0;
    long long end_to_end_us = 0;
};

struct ReferenceData {
    float q0sqr = 0.0f;
    std::vector<float> sums;
    std::vector<float> sums2;
    std::vector<float> c;
    std::vector<float> dN;
    std::vector<float> dS;
    std::vector<float> dW;
    std::vector<float> dE;
    std::vector<float> j_next;
};

long long elapsed_us(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
}

void print_stage_timing(const char* name,
                        long long in_us,
                        long long compute_us,
                        long long out_us) {
    std::printf("%s timing us: plio_in=%lld compute=%lld plio_out=%lld total=%lld\n",
                name, in_us, compute_us, out_us, in_us + compute_us + out_us);
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

    float v = 0.0f;
    int cnt = 0;
    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            if (!(fin >> v)) {
                std::fprintf(stderr,
                             "[warn] %s element count mismatch: got %d, expect %d\n",
                             path.c_str(), cnt, srad_cfg::kPixels);
                return false;
            }
            buf[srad_math::image_index(r, c)] = v;
            ++cnt;
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

float compute_q0sqr_from_stats(float sum, float sum2) {
    const float meanROI = sum / static_cast<float>(srad_cfg::kPixels);
    const float meanROI2 = meanROI * meanROI;
    const float varROI =
        (sum2 / static_cast<float>(srad_cfg::kPixels)) - meanROI2;
    return varROI / meanROI2;
}

struct OpenClReduceStats {
    float sum = 0.0f;
    float sum2 = 0.0f;
};

int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

int last_opencl_block_count(int no, int grid_dim) {
    return srad_cfg::kOpenClReductionThreads -
           (grid_dim * srad_cfg::kOpenClReductionThreads - no);
}

int largest_opencl_reduction_pow2(int n) {
    int df = 1;
    for (int i = 2; i <= srad_cfg::kOpenClReductionThreads; i *= 2) {
        if (n >= i) {
            df = i;
        }
    }
    return df;
}

void reduce_full_opencl_block(float* psum, float* psum2) {
    for (int i = 2; i <= srad_cfg::kOpenClReductionThreads; i *= 2) {
        for (int tx = i - 1; tx < srad_cfg::kOpenClReductionThreads; tx += i) {
            psum[tx] = psum[tx] + psum[tx - i / 2];
            psum2[tx] = psum2[tx] + psum2[tx - i / 2];
        }
    }
}

OpenClReduceStats opencl_v0_reduce_stats_reference(
    const std::vector<float>& image) {
    constexpr int kThreads = srad_cfg::kOpenClReductionThreads;
    std::vector<float> sums(srad_cfg::kPixels, 0.0f);
    std::vector<float> sums2(srad_cfg::kPixels, 0.0f);

    for (int i = 0; i < srad_cfg::kPixels; ++i) {
        const float v = image[i];
        sums[i] = v;
        sums2[i] = v * v;
    }

    int no = srad_cfg::kPixels;
    int mul = 1;
    int grid_dim = ceil_div_int(no, kThreads);

    while (grid_dim != 0) {
        const int nf = last_opencl_block_count(no, grid_dim);

        for (int bx = 0; bx < grid_dim; ++bx) {
            float psum[kThreads] = {};
            float psum2[kThreads] = {};

            for (int tx = 0; tx < kThreads; ++tx) {
                const int ei = bx * kThreads + tx;
                if (ei < no) {
                    const int src = ei * mul;
                    psum[tx] = sums[src];
                    psum2[tx] = sums2[src];
                }
            }

            const int dst = bx * mul * kThreads;
            if (nf == kThreads || bx != grid_dim - 1) {
                reduce_full_opencl_block(psum, psum2);
                sums[dst] = psum[kThreads - 1];
                sums2[dst] = psum2[kThreads - 1];
            } else {
                const int df = largest_opencl_reduction_pow2(nf);
                for (int i = 2; i <= df; i *= 2) {
                    for (int tx = i - 1; tx < df; tx += i) {
                        psum[tx] = psum[tx] + psum[tx - i / 2];
                        psum2[tx] = psum2[tx] + psum2[tx - i / 2];
                    }
                }

                const int tx = df - 1;
                for (int i = bx * kThreads + df; i < bx * kThreads + nf; ++i) {
                    psum[tx] = psum[tx] + sums[i];
                    psum2[tx] = psum2[tx] + sums2[i];
                }

                sums[dst] = psum[tx];
                sums2[dst] = psum2[tx];
            }
        }

        no = grid_dim;
        if (grid_dim == 1) {
            grid_dim = 0;
        } else {
            mul *= kThreads;
            grid_dim = ceil_div_int(no, kThreads);
        }
    }

    return {sums[0], sums2[0]};
}

float compute_q0sqr_reference(const std::vector<float>& image) {
    const OpenClReduceStats stats = opencl_v0_reduce_stats_reference(image);
    return compute_q0sqr_from_stats(stats.sum, stats.sum2);
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
    ref.sums.assign(srad_cfg::kPixels, 0.0f);
    ref.sums2.assign(srad_cfg::kPixels, 0.0f);
    ref.c.assign(srad_cfg::kPixels, 0.0f);
    ref.dN.assign(srad_cfg::kPixels, 0.0f);
    ref.dS.assign(srad_cfg::kPixels, 0.0f);
    ref.dW.assign(srad_cfg::kPixels, 0.0f);
    ref.dE.assign(srad_cfg::kPixels, 0.0f);
    ref.j_next.assign(srad_cfg::kPixels, 0.0f);

    for (int idx = 0; idx < srad_cfg::kPixels; ++idx) {
        ref.sums[idx] = image[idx];
        ref.sums2[idx] = image[idx] * image[idx];
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            const int idx = srad_math::image_index(i, j);
            const int iN = srad_math::north_row(i);
            const int iS = srad_math::south_row(i);
            const int jW = srad_math::west_col(j);
            const int jE = srad_math::east_col(j);

            const float JC = image[idx];
            const float JN = image[srad_math::image_index(iN, j)];
            const float JS = image[srad_math::image_index(iS, j)];
            const float JW = image[srad_math::image_index(i, jW)];
            const float JE = image[srad_math::image_index(i, jE)];

            const float dN = JN - JC;
            const float dS = JS - JC;
            const float dW = JW - JC;
            const float dE = JE - JC;

            ref.dN[idx] = dN;
            ref.dS[idx] = dS;
            ref.dW[idx] = dW;
            ref.dE[idx] = dE;
            ref.c[idx] = compute_c_reference(JC, dN, dS, dW, dE, ref.q0sqr);
        }
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            const int idx = srad_math::image_index(i, j);
            const int iS = srad_math::south_row(i);
            const int jE = srad_math::east_col(j);
            const int south_idx = srad_math::image_index(iS, j);
            const int east_idx = srad_math::image_index(i, jE);

            const float D =
                ref.c[idx] * ref.dN[idx] +
                ref.c[south_idx] * ref.dS[idx] +
                ref.c[idx] * ref.dW[idx] +
                ref.c[east_idx] * ref.dE[idx];
            ref.j_next[idx] = image[idx] + 0.25f * lambda * D;
        }
    }

    return ref;
}

void compare_array(const char* name,
                   const float* got,
                   const std::vector<float>& gold,
                   int n) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int mismatch_count = 0;

    for (int i = 0; i < n; ++i) {
        const float abs_err = std::fabs(got[i] - gold[i]);
        const float denom = std::max(std::fabs(gold[i]), 1.0e-12f);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, abs_err / denom);
        if (abs_err > kCompareTol) ++mismatch_count;
    }

    std::printf("%s max_abs_error_float: %.9g max_relative_error: %.9g mismatch_count_tol_%g: %d\n",
                name, max_abs, max_rel, kCompareTol, mismatch_count);
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

int tile_count_rows() {
    return srad_cfg::kTileRows;
}

int tile_count_cols() {
    return srad_cfg::kTileCols;
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
                     "[error] this faithful CUDA baseline supports one iteration; got %d\n",
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
    std::printf("lambda float32        : %.9g\n", lambda);
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

    auto load_prepare = open_kernel_or_die(device, xrt_uuid, "LoadPrepare");
    auto store_prepare = open_kernel_or_die(device, xrt_uuid, "StorePrepare");
    auto load_reduce = open_kernel_or_die(device, xrt_uuid, "LoadReduce");
    auto store_reduce = open_kernel_or_die(device, xrt_uuid, "StoreReduce");
    auto load_coeff = open_kernel_or_die(device, xrt_uuid, "LoadCoeff");
    auto store_coeff = open_kernel_or_die(device, xrt_uuid, "StoreCoeff");
    auto load_update = open_kernel_or_die(device, xrt_uuid, "LoadUpdate");
    auto store_update = open_kernel_or_die(device, xrt_uuid, "StoreUpdate");

    auto image_bo = xrt::bo(device, srad_cfg::kImageBytes, load_prepare.group_id(0));
    auto sums_bo =
        xrt::bo(device, srad_cfg::kImageBytes, store_prepare.group_id(0));
    auto sums2_bo =
        xrt::bo(device, srad_cfg::kImageBytes, store_prepare.group_id(1));
    auto stats_bo =
        xrt::bo(device, srad_cfg::kScalarPacketBytes, store_reduce.group_id(0));
    auto c_bo = xrt::bo(device, srad_cfg::kImageBytes, store_coeff.group_id(0));
    auto dN_bo = xrt::bo(device, srad_cfg::kImageBytes, store_coeff.group_id(1));
    auto dS_bo = xrt::bo(device, srad_cfg::kImageBytes, store_coeff.group_id(2));
    auto dW_bo = xrt::bo(device, srad_cfg::kImageBytes, store_coeff.group_id(3));
    auto dE_bo = xrt::bo(device, srad_cfg::kImageBytes, store_coeff.group_id(4));
    auto out_bo = xrt::bo(device, srad_cfg::kImageBytes, store_update.group_id(0));

    auto image_map = image_bo.map<float*>();
    auto sums_map = sums_bo.map<float*>();
    auto sums2_map = sums2_bo.map<float*>();
    auto stats_map = stats_bo.map<float*>();
    auto c_map = c_bo.map<float*>();
    auto dN_map = dN_bo.map<float*>();
    auto dS_map = dS_bo.map<float*>();
    auto dW_map = dW_bo.map<float*>();
    auto dE_map = dE_bo.map<float*>();
    auto out_map = out_bo.map<float*>();

    std::memcpy(image_map, image.data(), srad_cfg::kImageBytes);
    std::memset(sums_map, 0, srad_cfg::kImageBytes);
    std::memset(sums2_map, 0, srad_cfg::kImageBytes);
    std::memset(stats_map, 0, srad_cfg::kScalarPacketBytes);
    std::memset(c_map, 0, srad_cfg::kImageBytes);
    std::memset(dN_map, 0, srad_cfg::kImageBytes);
    std::memset(dS_map, 0, srad_cfg::kImageBytes);
    std::memset(dW_map, 0, srad_cfg::kImageBytes);
    std::memset(dE_map, 0, srad_cfg::kImageBytes);
    std::memset(out_map, 0, srad_cfg::kImageBytes);

    image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    sums_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    sums2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    stats_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    dN_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    dS_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    dW_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    dE_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const int n_tile_rows = tile_count_rows();
    const int n_tile_cols = tile_count_cols();
    const int n_tiles = n_tile_rows * n_tile_cols;
    PipelineTiming timing;

    const auto t0 = Clock::now();

    std::printf("---- GraphPrepare: DDR(d_I) -> PL -> AIE(d_sums/d_sums2) -> PL -> DDR ----\n");
    std::fflush(stdout);
    graphPrepare.init();
    std::printf("[host] GraphPrepare full image\n");
    std::fflush(stdout);
    auto stage_t0 = Clock::now();
    auto prepare_sink_run = store_prepare(sums_bo, sums2_bo);
    graphPrepare.run(1);
    auto prepare_source_run = load_prepare(image_bo);
    prepare_source_run.wait();
    timing.prepare_in_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    graphPrepare.wait();
    timing.prepare_compute_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    prepare_sink_run.wait();
    timing.prepare_out_us += elapsed_us(stage_t0, Clock::now());
    graphPrepare.end();

    sums_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    sums2_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    compare_array("d_sums ", sums_map, ref.sums, srad_cfg::kPixels);
    compare_array("d_sums2", sums2_map, ref.sums2, srad_cfg::kPixels);

    std::printf("---- GraphReduce: DDR(d_sums/d_sums2) -> PL -> AIE(reduce) -> PL -> DDR ----\n");
    std::fflush(stdout);
    graphReduce.init();
    std::printf("[host] GraphReduce full d_sums/d_sums2 arrays\n");
    std::fflush(stdout);
    stage_t0 = Clock::now();
    auto reduce_sink_run = store_reduce(stats_bo);
    graphReduce.run(1);
    auto source_run = load_reduce(sums_bo, sums2_bo);
    source_run.wait();
    timing.reduce_in_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    graphReduce.wait();
    timing.reduce_compute_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    reduce_sink_run.wait();
    timing.reduce_out_us += elapsed_us(stage_t0, Clock::now());
    graphReduce.end();

    stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const float reduce_sum = stats_map[0];
    const float reduce_sum2 = stats_map[1];
    const float q0sqr_host = compute_q0sqr_from_stats(reduce_sum, reduce_sum2);
    std::printf("reduce_sum float32 : %.9g\n", reduce_sum);
    std::printf("reduce_sum2 float32: %.9g\n", reduce_sum2);
    std::printf("q0sqr_ref float32  : %.9g\n", ref.q0sqr);
    std::printf("q0sqr_host float32 : %.9g\n", q0sqr_host);
    std::printf("q0_abs_error_float: %.9g\n",
                std::fabs(ref.q0sqr - q0sqr_host));

    std::printf("---- GraphCoeff: DDR(J) + host q0 scalar -> PL -> AIE -> PL -> DDR ----\n");
    std::fflush(stdout);
    graphCoeff.init();
    std::printf("[host] GraphCoeff %d tiles\n", n_tiles);
    std::fflush(stdout);
    stage_t0 = Clock::now();
    auto coeff_sink_run = store_coeff(c_bo, dN_bo, dS_bo, dW_bo, dE_bo);
    graphCoeff.run(n_tiles);
    auto coeff_source_run = load_coeff(image_bo, q0sqr_host);
    coeff_source_run.wait();
    timing.coeff_in_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    graphCoeff.wait();
    timing.coeff_compute_us += elapsed_us(stage_t0, Clock::now());

    stage_t0 = Clock::now();
    coeff_sink_run.wait();
    timing.coeff_out_us += elapsed_us(stage_t0, Clock::now());
    graphCoeff.end();

    c_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dN_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dS_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dW_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    dE_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    compare_array("d_c ", c_map, ref.c, srad_cfg::kPixels);
    compare_array("d_dN", dN_map, ref.dN, srad_cfg::kPixels);
    compare_array("d_dS", dS_map, ref.dS, srad_cfg::kPixels);
    compare_array("d_dW", dW_map, ref.dW, srad_cfg::kPixels);
    compare_array("d_dE", dE_map, ref.dE, srad_cfg::kPixels);

    std::printf("---- GraphUpdate: AIE-K2 tiled reread of J and five arrays ----\n");
    std::fflush(stdout);
    graphUpdate.init();
    std::printf("[host] GraphUpdate %d tiles\n", n_tiles);
    std::fflush(stdout);
    stage_t0 = Clock::now();
    auto sink_run = store_update(out_bo);
    graphUpdate.run(n_tiles);
    auto update_source_run = load_update(image_bo,
                                         c_bo,
                                         dN_bo,
                                         dS_bo,
                                         dW_bo,
                                         dE_bo,
                                         lambda);
    update_source_run.wait();
    timing.update_in_us += elapsed_us(stage_t0, Clock::now());
    stage_t0 = Clock::now();
    graphUpdate.wait();
    timing.update_compute_us += elapsed_us(stage_t0, Clock::now());
    stage_t0 = Clock::now();
    sink_run.wait();
    timing.update_out_us += elapsed_us(stage_t0, Clock::now());
    graphUpdate.end();

    const auto t1 = Clock::now();
    const auto dur_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<float> got(out_map, out_map + srad_cfg::kPixels);
    print_preview("output preview:", got);
    compare_output(got, ref.j_next);
    print_stage_timing("GraphPrepare", timing.prepare_in_us,
                       timing.prepare_compute_us, timing.prepare_out_us);
    print_stage_timing("GraphReduce ", timing.reduce_in_us,
                       timing.reduce_compute_us, timing.reduce_out_us);
    print_stage_timing("GraphCoeff ", timing.coeff_in_us,
                       timing.coeff_compute_us, timing.coeff_out_us);
    print_stage_timing("GraphUpdate", timing.update_in_us,
                       timing.update_compute_us, timing.update_out_us);
    std::printf("staged end-to-end time: %lld us\n", static_cast<long long>(dur_us));

    CommonTiming common;
    common.data_transfer_us =
        timing.prepare_in_us + timing.prepare_out_us +
        timing.reduce_in_us + timing.reduce_out_us +
        timing.coeff_in_us + timing.coeff_out_us +
        timing.update_in_us + timing.update_out_us;
    common.logic_compute_us =
        timing.prepare_compute_us +
        timing.reduce_compute_us +
        timing.coeff_compute_us +
        timing.update_compute_us;
    common.end_to_end_us = dur_us;
    print_common_timing("CUDA", common);

    dump_float_file(output_path, got);

    // Keep the explicit C XRT handle open until process exit. In Vitis 2023.2
    // hw_emu, closing it manually after mixed C++ XRT PL objects and ADF can
    // trigger a zocl cleanup crash after valid results print.
    (void)dhdl;

    return EXIT_SUCCESS;
}
