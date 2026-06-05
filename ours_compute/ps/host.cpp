// Ours-compute SRAD mapping:
//   PL TopPL computes global q0sqr.
//   AIE K1 srad_local_q computes local directional differences.
//   AIE K2 srad_coeff_update uses the original five-division c expression
//   and updates J.
//
// The host never materializes d_c/dN/dS/dW/dE and never feeds host-computed
// q0sqr to AIE in the formal path. q0sqr_ref is CPU-side validation only;
// AIE receives the PL-computed q0 packet through ordinary GMIO.

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

GraphOursPLQ0 graphOursPLQ0("ours_compute_plq0");

namespace {

constexpr int PREVIEW = 16;
constexpr float kCompareTol = 1.0e-5f;
using Clock = std::chrono::high_resolution_clock;

struct PipelineTiming {
    long long q0_bo_to_device_us = 0;
    long long graph_output_submit_us = 0;
    long long graph_input_us = 0;
    long long q0_pl_compute_us = 0;
    long long pl_aie_compute_us = 0;
    long long graph_output_us = 0;
    long long q0_debug_from_device_us = 0;
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
    std::printf("---- Common timing summary (host wall us): Ours-compute ----\n");
    std::printf("data_transfer_us: %lld\n", timing.data_transfer_us);
    std::printf("logic_compute_us: %lld\n", timing.logic_compute_us);
    std::printf("end_to_end_us   : %lld\n", timing.end_to_end_us);
}

void print_stage_timing(const char* name,
                        long long in_us,
                        long long compute_us,
                        long long out_us) {
    std::printf("%s timing us: gm2aie=%lld compute=%lld aie2gm=%lld total=%lld\n",
                name,
                in_us,
                compute_us,
                out_us,
                in_us + compute_us + out_us);
}

void print_toppl_timing(const PipelineTiming& timing) {
    const long long total =
        timing.q0_bo_to_device_us + timing.q0_pl_compute_us +
        timing.q0_debug_from_device_us;
    std::printf("TopPL q0 timing us: h2d=%lld compute=%lld d2h=%lld total=%lld\n",
                timing.q0_bo_to_device_us,
                timing.q0_pl_compute_us,
                timing.q0_debug_from_device_us,
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
    for (int r = 0; r < ROW; ++r) {
        for (int c = 0; c < COL; ++c) {
            image[r * COL + c] =
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
            fout << buf[r * COL + c];
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

void pack_8x8_blocks(const std::vector<float>& image,
                     std::vector<float>& blocks) {
    int out = 0;
    for (int br = 0; br < ROW; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < COL; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                const int src = (br + r) * COL + bc;
                std::memcpy(blocks.data() + out,
                            image.data() + src,
                            srad_cfg::kBlockCols * sizeof(float));
                out += srad_cfg::kBlockCols;
            }
        }
    }
}

void unpack_8x8_blocks(const std::vector<float>& blocks,
                       std::vector<float>& image) {
    int in = 0;
    for (int br = 0; br < ROW; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < COL; bc += srad_cfg::kBlockCols) {
            for (int r = 0; r < srad_cfg::kBlockRows; ++r) {
                const int dst = (br + r) * COL + bc;
                std::memcpy(image.data() + dst,
                            blocks.data() + in,
                            srad_cfg::kBlockCols * sizeof(float));
                in += srad_cfg::kBlockCols;
            }
        }
    }
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

    const float g2 =
        (dN * dN + dS * dS + dW * dW + dE * dE) / (jc * jc);
    const float lap = (dN + dS + dW + dE) / jc;
    const float num =
        srad_math::kHalf * g2 -
        srad_math::kOneSixteenth * lap * lap;
    const float den = srad_math::kOne + srad_math::kQuarter * lap;
    const float qsqr = num / (den * den);
    const float ratio =
        (qsqr - q0sqr) / (q0sqr * (srad_math::kOne + q0sqr));
    const float c = srad_math::kOne / (srad_math::kOne + ratio);

    return srad_math::clamp01(c);
}

ReferenceData cpu_reference(const std::vector<float>& image, float lambda) {
    ReferenceData ref;
    ref.q0sqr = compute_q0sqr_reference(image);
    ref.j_next.assign(srad_cfg::kPixels, 0.0f);

    std::vector<float> c_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dN_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dS_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dW_plane(srad_cfg::kPixels, 0.0f);
    std::vector<float> dE_plane(srad_cfg::kPixels, 0.0f);

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

            dN_plane[idx] = dN;
            dS_plane[idx] = dS;
            dW_plane[idx] = dW;
            dE_plane[idx] = dE;
            c_plane[idx] = compute_c_reference(JC, dN, dS, dW, dE, ref.q0sqr);
        }
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            const int idx = srad_math::image_index(i, j);
            const int south_idx =
                srad_math::image_index(srad_math::south_row(i), j);
            const int east_idx =
                srad_math::image_index(i, srad_math::east_col(j));

            const float D =
                c_plane[idx] * dN_plane[idx] +
                c_plane[south_idx] * dS_plane[idx] +
                c_plane[idx] * dW_plane[idx] +
                c_plane[east_idx] * dE_plane[idx];

            ref.j_next[idx] =
                image[idx] + srad_math::kQuarter * lambda * D;
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

template <typename T>
T* gmio_malloc_or_die(std::size_t bytes, const char* name) {
    auto* ptr = reinterpret_cast<T*>(adf::GMIO::malloc(bytes));
    if (!ptr) {
        std::fprintf(stderr, "[error] GMIO::malloc failed for %s\n", name);
        std::exit(EXIT_FAILURE);
    }
    return ptr;
}

xrt::kernel open_toppl_kernel(const xrt::device& device,
                              const xrt::uuid& uuid) {
    try {
        return xrt::kernel(device, uuid, "TopPL:{TopPL_1}");
    } catch (const std::exception&) {
        return xrt::kernel(device, uuid, "TopPL");
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
        (argc >= 4) ? argv[3] : "./data/input_32x32.txt";
    const std::string output_path =
        (argc >= 5) ? argv[4] : "./data/aie_j_next.txt";
    const float lambda = (argc >= 6) ? static_cast<float>(std::atof(argv[5]))
                                     : srad_cfg::kLambdaDefault;

    if (iter_cnt != 1) {
        std::fprintf(stderr,
                     "[error] Ours currently supports one iteration; got %d\n",
                     iter_cnt);
        return EXIT_FAILURE;
    }

    std::vector<float> image(srad_cfg::kPixels);
    if (!load_float_file(input_path, image)) {
        std::fprintf(stderr, "[warn] fallback to deterministic input\n");
        fill_default_image(image);
    }
    std::vector<float> image_blocks(srad_cfg::kPixels);
    pack_8x8_blocks(image, image_blocks);

    const ReferenceData ref = cpu_reference(image, lambda);

    std::printf("mapping               : Ours-compute PL-q0 + AIE K1/K2 original-compute pipeline\n");
    std::printf("trace kernels         : srad_local_q, srad_coeff_update_original\n");
    std::printf("image size            : %dx%d (%d pixels)\n",
                ROW, COL, srad_cfg::kPixels);
    std::printf("block schedule        : %d blocks, %dx%d (%d floats) per graph firing\n",
                srad_cfg::kBlockCount,
                srad_cfg::kBlockRows,
                srad_cfg::kBlockCols,
                srad_cfg::kBlockPixels);
    std::printf("data size bytes       : J=%d J_next=%d mid_buffer_total=%d\n",
                srad_cfg::kImageBytes,
                srad_cfg::kImageBytes,
                srad_cfg::kMidBytes);
    std::printf("float lanes per packet: %d\n", srad_cfg::kLanes);
    std::printf("lambda float32        : %.9g\n", lambda);
    std::printf("q0sqr_ref float32     : %.9g\n", ref.q0sqr);
    print_preview("input preview:", image);

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

        auto q0_kernel = open_toppl_kernel(device, xrt_uuid);
        auto q0_input_bo =
            xrt::bo(device, srad_cfg::kImageBytes, q0_kernel.group_id(0));
        auto q0_debug_bo =
            xrt::bo(device,
                    srad_cfg::kScalarPacketBytes,
                    q0_kernel.group_id(1));
        auto q0_input_map = q0_input_bo.map<float*>();
        auto q0_debug_map = q0_debug_bo.map<float*>();

        std::memcpy(q0_input_map, image.data(), srad_cfg::kImageBytes);
        std::memset(q0_debug_map, 0, srad_cfg::kScalarPacketBytes);

        auto* j_buf = gmio_malloc_or_die<float>(srad_cfg::kImageBytes, "J");
        auto* q0_buf =
            gmio_malloc_or_die<float>(srad_cfg::kScalarPacketsPerImageBytes,
                                      "q0sqr");
        auto* lambda_buf =
            gmio_malloc_or_die<float>(srad_cfg::kScalarPacketsPerImageBytes,
                                      "lambda");
        auto* out_buf =
            gmio_malloc_or_die<float>(srad_cfg::kImageBytes, "J_next");

        std::memcpy(j_buf, image_blocks.data(), srad_cfg::kImageBytes);
        std::memset(q0_buf, 0, srad_cfg::kScalarPacketsPerImageBytes);
        std::memset(lambda_buf, 0, srad_cfg::kScalarPacketsPerImageBytes);
        for (int blk = 0; blk < srad_cfg::kBlockCount; ++blk) {
            lambda_buf[blk * srad_cfg::kScalarPacketElems] = lambda;
        }
        std::memset(out_buf, 0, srad_cfg::kImageBytes);

        PipelineTiming timing;
        const auto t0 = Clock::now();

        auto stage_t0 = Clock::now();
        q0_input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        q0_debug_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.q0_bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        auto q0_run = q0_kernel(q0_input_bo, q0_debug_bo);
        q0_run.wait();
        timing.q0_pl_compute_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        q0_debug_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        timing.q0_debug_from_device_us = elapsed_us(stage_t0, Clock::now());

        const float q0sqr_pl_for_aie = q0_debug_map[0];
        for (int blk = 0; blk < srad_cfg::kBlockCount; ++blk) {
            q0_buf[blk * srad_cfg::kScalarPacketElems] = q0sqr_pl_for_aie;
        }

        std::printf("---- PL TopPL + GraphOursComputePLQ0 ----\n");
        std::printf("PL q0 input           : J_reduce float[%d]\n",
                    srad_cfg::kPixels);
        std::printf("PL q0 output          : q0_debug DDR scalar packet\n");
        std::printf("AIE q0 input          : GMIO scalar packet from PL q0_debug\n");
        std::printf("AIE block firing      : graph.run(%d), %d floats per firing\n",
                    srad_cfg::kBlockCount,
                    srad_cfg::kBlockPixels);
        std::printf("AIE K1                : srad_local_q local differences\n");
        std::printf("AIE K2                : srad_coeff_update original five-division c\n");
        std::fflush(stdout);

        graphOursPLQ0.init();

        stage_t0 = Clock::now();
        graphOursPLQ0.out_j_next.aie2gm_nb(out_buf, srad_cfg::kImageBytes);
        timing.graph_output_submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        graphOursPLQ0.in_j.gm2aie_nb(j_buf, srad_cfg::kImageBytes);
        graphOursPLQ0.in_q0sqr.gm2aie_nb(q0_buf,
                                          srad_cfg::kScalarPacketsPerImageBytes);
        graphOursPLQ0.in_lambda.gm2aie_nb(lambda_buf,
                                          srad_cfg::kScalarPacketsPerImageBytes);
        graphOursPLQ0.in_j.wait();
        graphOursPLQ0.in_q0sqr.wait();
        graphOursPLQ0.in_lambda.wait();
        timing.graph_input_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        graphOursPLQ0.run(srad_cfg::kBlockCount);
        graphOursPLQ0.wait();
        timing.pl_aie_compute_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        graphOursPLQ0.out_j_next.wait();
        timing.graph_output_us = elapsed_us(stage_t0, Clock::now());

        graphOursPLQ0.end();

        const auto t1 = Clock::now();
        const long long dur_us = elapsed_us(t0, t1);

        const float q0sqr_pl = q0_debug_map[0];
        std::printf("q0sqr_pl float32      : %.9g\n", q0sqr_pl);
        std::printf("q0_abs_error_float    : %.9g\n",
                    std::fabs(ref.q0sqr - q0sqr_pl));

        std::vector<float> got_blocks(srad_cfg::kPixels);
        std::memcpy(got_blocks.data(), out_buf, srad_cfg::kImageBytes);
        std::vector<float> got(srad_cfg::kPixels);
        unpack_8x8_blocks(got_blocks, got);
        print_preview("output preview:", got);
        compare_output(got, ref.j_next);

        print_toppl_timing(timing);
        print_stage_timing("GraphOursCompute",
                           timing.graph_input_us,
                           timing.pl_aie_compute_us,
                           timing.graph_output_submit_us +
                               timing.graph_output_us);
        std::printf("staged graph/PL time: %lld us\n", dur_us);

        std::printf("timing us: q0_h2d=%lld q0_pl_compute=%lld q0_debug_d2h=%lld graph_h2aie=%lld output_submit=%lld aie_compute=%lld graph_aie2h=%lld total=%lld\n",
                    timing.q0_bo_to_device_us,
                    timing.q0_pl_compute_us,
                    timing.q0_debug_from_device_us,
                    timing.graph_input_us,
                    timing.graph_output_submit_us,
                    timing.pl_aie_compute_us,
                    timing.graph_output_us,
                    dur_us);

        CommonTiming common;
        common.data_transfer_us =
            timing.q0_bo_to_device_us +
            timing.graph_output_submit_us +
            timing.graph_input_us +
            timing.graph_output_us +
            timing.q0_debug_from_device_us;
        common.logic_compute_us =
            timing.q0_pl_compute_us + timing.pl_aie_compute_us;
        common.end_to_end_us = dur_us;
        print_common_timing(common);

        dump_float_file(output_path, got);

        adf::GMIO::free(j_buf);
        adf::GMIO::free(q0_buf);
        adf::GMIO::free(lambda_buf);
        adf::GMIO::free(out_buf);
        xrtDeviceClose(dhdl);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
