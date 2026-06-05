// Ours-undivide SRAD mapping:
//   PL TopPL computes global q0sqr.
//   AIE fused pressure kernel mirrors current ours K1/K2 logic in 16x16
//   fused tiles with a single-row direct kernel and no manual spill.
//   It computes local C and then updates each row without
//   materializing the mid plane.
//
// The host never materializes d_c/dN/dS/dW/dE and never feeds host-computed
// q0sqr to AIE in the formal path. q0sqr_ref is CPU-side validation only;
// AIE receives the PL-computed q0 packet through PLIO.

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

GraphOursPLQ0 graphOursPLQ0("ours_undivide_plq0");

namespace {

constexpr int PREVIEW = 16;
constexpr float kCompareTol = 1.0e-5f;
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

struct ReferenceData {
    float q0sqr = 0.0f;
    std::vector<float> j_next;
};

long long elapsed_us(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
        .count();
}

void print_common_timing(const CommonTiming& timing) {
    std::printf("---- Common timing summary (host wall us): Ours-undivide ----\n");
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

float update_j_reference(float jc, float d, float lambda) {
    return jc + srad_math::kQuarter * lambda * d;
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

    for (int br = 0; br < ROW; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < COL; bc += srad_cfg::kBlockCols) {
            for (int tr = 0; tr < srad_cfg::kBlockRows; ++tr) {
                for (int tc = 0; tc < srad_cfg::kBlockCols; ++tc) {
                    const int i = br + tr;
                    const int j = bc + tc;
                    const int idx = srad_math::image_index(i, j);
                    const int trN = (tr == 0) ? 0 : tr - 1;
                    const int trS = (tr == srad_cfg::kBlockRows - 1)
                                        ? srad_cfg::kBlockRows - 1
                                        : tr + 1;
                    const int tcW = (tc == 0) ? 0 : tc - 1;
                    const int tcE = (tc == srad_cfg::kBlockCols - 1)
                                        ? srad_cfg::kBlockCols - 1
                                        : tc + 1;

                    const float JC = image[idx];
                    const float dN =
                        image[srad_math::image_index(br + trN, j)] - JC;
                    const float dS =
                        image[srad_math::image_index(br + trS, j)] - JC;
                    const float dW =
                        image[srad_math::image_index(i, bc + tcW)] - JC;
                    const float dE =
                        image[srad_math::image_index(i, bc + tcE)] - JC;

                    dN_plane[idx] = dN;
                    dS_plane[idx] = dS;
                    dW_plane[idx] = dW;
                    dE_plane[idx] = dE;
                    c_plane[idx] =
                        compute_c_reference(JC, dN, dS, dW, dE, ref.q0sqr);
                }
            }
        }
    }

    for (int br = 0; br < ROW; br += srad_cfg::kBlockRows) {
        for (int bc = 0; bc < COL; bc += srad_cfg::kBlockCols) {
            for (int tr = 0; tr < srad_cfg::kBlockRows; ++tr) {
                for (int tc = 0; tc < srad_cfg::kBlockCols; ++tc) {
                    const int i = br + tr;
                    const int j = bc + tc;
                    const int idx = srad_math::image_index(i, j);
                    const int trS = (tr == srad_cfg::kBlockRows - 1)
                                        ? srad_cfg::kBlockRows - 1
                                        : tr + 1;
                    const int tcE = (tc == srad_cfg::kBlockCols - 1)
                                        ? srad_cfg::kBlockCols - 1
                                        : tc + 1;
                    const int south_idx =
                        srad_math::image_index(br + trS, j);
                    const int east_idx =
                        srad_math::image_index(i, bc + tcE);

                    const float D =
                        c_plane[idx] * dN_plane[idx] +
                        c_plane[south_idx] * dS_plane[idx] +
                        c_plane[idx] * dW_plane[idx] +
                        c_plane[east_idx] * dE_plane[idx];

                    ref.j_next[idx] = update_j_reference(image[idx], D, lambda);
                }
            }
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
    const ReferenceData ref = cpu_reference(image, lambda);

    std::printf("mapping               : fused pressure 16x16 single-kernel tile on 16x16 data\n");
    std::printf("stack experiment      : 16x16 data, 16x16 tile, single-row direct fused kernel, compiler spill only, full 256-point update per firing\n");
    std::printf("trace kernels         : srad_undivide_fused\n");
    std::printf("image size            : %dx%d (%d pixels)\n",
                ROW, COL, srad_cfg::kPixels);
    std::printf("block schedule        : %d blocks, %dx%d (%d floats) per graph firing\n",
                srad_cfg::kBlockCount,
                srad_cfg::kBlockRows,
                srad_cfg::kBlockCols,
                srad_cfg::kBlockPixels);
    std::printf("data size bytes       : J=%d J_next=%d mid_buffer_total=0 (fused)\n",
                srad_cfg::kImageBytes,
                srad_cfg::kImageBytes);
    std::printf("float lanes per packet: %d\n", srad_cfg::kLanes);
    std::printf("update formula        : J_next = J + 0.25 * lambda * D\n");
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

        auto toppl_kernel = open_toppl_kernel(device, xrt_uuid);
        auto input_bo =
            xrt::bo(device, srad_cfg::kImageBytes, toppl_kernel.group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kImageBytes, toppl_kernel.group_id(1));
        auto input_map = input_bo.map<float*>();
        auto output_map = output_bo.map<float*>();

        std::memcpy(input_map, image.data(), srad_cfg::kImageBytes);
        std::memset(output_map, 0, srad_cfg::kImageBytes);

        PipelineTiming timing;
        const auto t0 = Clock::now();

        auto stage_t0 = Clock::now();
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing.bo_to_device_us = elapsed_us(stage_t0, Clock::now());

        std::printf("---- PL TopPL + GraphOursUndividePLQ0 ----\n");
        std::printf("PL input              : DDR J float[%d]\n",
                    srad_cfg::kPixels);
        std::printf("PL q0/output packing  : q0sqr + J + J_update + lambda to AIE PLIO\n");
        std::printf("AIE q0 input          : PLIO scalar packet from TopPL\n");
        std::printf("AIE block firing      : graph.run(%d), %d floats per firing\n",
                    srad_cfg::kBlockCount,
                    srad_cfg::kBlockPixels);
        std::printf("AIE SIMD granularity  : v8float handles %d floats per vector instruction; each 16x16 firing contains %d vector chunks\n",
                    srad_cfg::kLanes,
                    srad_cfg::kVectorLoopsPerBlock);
        std::printf("AIE pressure group    : %d rows, %d v8float vectors per inner group\n",
                    srad_cfg::kPressureRowsPerGroup,
                    srad_cfg::kPressureVectorsPerGroup);
        std::printf("AIE J inputs          : in_j for coefficient pass, in_j_update for update pass\n");
        std::printf("AIE fused             : srad_undivide_fused computes C and updates one row at a time with no manual spill while one graph firing still covers all 256 points\n");
        std::printf("AIE update            : standard J_next = J + 0.25 * lambda * D\n");
        std::fflush(stdout);

        graphOursPLQ0.init();

        stage_t0 = Clock::now();
        graphOursPLQ0.run(srad_cfg::kBlockCount);
        auto toppl_run = toppl_kernel(input_bo, output_bo, lambda);
        timing.submit_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        toppl_run.wait();
        timing.toppl_wait_us = elapsed_us(stage_t0, Clock::now());

        stage_t0 = Clock::now();
        graphOursPLQ0.wait();
        timing.graph_wait_us = elapsed_us(stage_t0, Clock::now());

        graphOursPLQ0.end();

        const auto t1 = Clock::now();
        const long long dur_us = elapsed_us(t0, t1);

        stage_t0 = Clock::now();
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        timing.bo_from_device_us = elapsed_us(stage_t0, Clock::now());
        std::vector<float> got(output_map, output_map + srad_cfg::kPixels);
        print_preview("output preview:", got);
        compare_output(got, ref.j_next);

        print_toppl_timing(timing);
        print_stage_timing("GraphOursUndivide",
                           timing.toppl_wait_us,
                           timing.graph_wait_us,
                           timing.bo_from_device_us);
        std::printf("staged graph/PL time: %lld us\n", dur_us);

        std::printf("timing us: h2d=%lld submit=%lld toppl_plio=%lld graph_wait=%lld d2h=%lld total=%lld\n",
                    timing.bo_to_device_us,
                    timing.submit_us,
                    timing.toppl_wait_us,
                    timing.graph_wait_us,
                    timing.bo_from_device_us,
                    dur_us);

        CommonTiming common;
        common.data_transfer_us =
            timing.bo_to_device_us + timing.bo_from_device_us;
        common.logic_compute_us =
            timing.toppl_wait_us + timing.graph_wait_us;
        common.end_to_end_us = dur_us;
        print_common_timing(common);

        dump_float_file(output_path, got);

        xrtDeviceClose(dhdl);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] XRT/AIE execution failed: %s\n", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
