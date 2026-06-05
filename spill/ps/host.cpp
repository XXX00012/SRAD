// Spill SRAD mapping:
//   PL TopPL computes global q0sqr.
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

bool is_compare_pixel(int idx) {
    (void)idx;
    return true;
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

    std::printf("mapping               : Spill PL-q0 + AIE K1 coeff package + K2 recomputed gradients\n");
    std::printf("trace kernels         : srad_local_q, srad_coeff_update\n");
    std::printf("image size            : %dx%d (%d pixels)\n",
                srad_cfg::kRows, srad_cfg::kCols, srad_cfg::kPixels);
    std::printf("tile schedule         : %dx%d output tiles, output %dx%d, input logical %dx%d, input physical row=%d; %d graph firing(s), %d input floats/firing, %d output floats/firing\n",
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

        auto toppl_kernel = open_toppl_kernel(device, xrt_uuid);
        auto input_bo =
            xrt::bo(device, srad_cfg::kImageBytes, toppl_kernel.group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kOutputBytes, toppl_kernel.group_id(1));
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

        std::printf("---- PL TopPL + GraphOursPLQ0 ----\n");
        std::printf("PL input              : DDR J float[%d]\n",
                    srad_cfg::kPixels);
        std::printf("PL q0/output packing  : q0sqr embedded in first padding column of each 19x24 J tile\n");
        std::printf("AIE q0 input          : K1 reads q0sqr from the J tile padding slot\n");
        std::printf("AIE tile firing       : graph.run(%d), %d input floats per firing, %d output floats per firing\n",
                    srad_cfg::kGraphRunIterations,
                    srad_cfg::kImageInputSampleElems,
                    srad_cfg::kOutputSampleElems);
        std::printf("PL output store       : one 16x16 result tile per firing into compact %dx%d output\n",
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

        graphOursPLQ0.init();

        stage_t0 = Clock::now();
        graphOursPLQ0.run(srad_cfg::kGraphRunIterations);
        auto toppl_run = toppl_kernel(input_bo, output_bo);
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
        std::vector<float> got(output_map, output_map + srad_cfg::kOutputElems);
        print_preview("output preview:", got);
        compare_output(got, ref.j_next);

        print_toppl_timing(timing);
        print_stage_timing("GraphOurs",
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
