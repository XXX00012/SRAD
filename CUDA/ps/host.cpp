#include "TopGraph.h"

#include <adf/adf_api/XRTConfig.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>
#include <xrt/xrt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

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

int active_blocks(int block_count) {
    int active = block_count;
    if (active < 1) {
        active = 1;
    }
    if (active > srad_cfg::kCudaBlocks) {
        active = srad_cfg::kCudaBlocks;
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
    size_t count = 0;
    while (count < buf.size() && (fin >> value)) {
        buf[count++] = value;
    }

    if (count != buf.size()) {
        std::fprintf(stderr,
                     "[error] %s element count mismatch: got %zu, expect %zu\n",
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

    for (int row = 0; row < srad_cfg::kBoardRows; ++row) {
        for (int col = 0; col < srad_cfg::kBoardCols; ++col) {
            if (col) {
                fout << ' ';
            }
            fout << output[row * srad_cfg::kBoardCols + col];
        }
        fout << '\n';
    }
}

xrt::kernel open_kernel(const xrt::device& device,
                        const xrt::uuid& uuid,
                        const char* name) {
    const std::string base(name);
    const std::string cu = base + "_0";
    try {
        return xrt::kernel(device, uuid, base + ":{" + cu + "}");
    } catch (const std::exception&) {
        try {
            return xrt::kernel(device, uuid, cu);
        } catch (const std::exception&) {
            return xrt::kernel(device, uuid, base);
        }
    }
}

float compute_q0sqr(const float* partials, int blocks) {
    double sum = 0.0;
    double sum2 = 0.0;
    for (int block = 0; block < blocks; ++block) {
        sum += partials[(2 * block) + 0];
        sum2 += partials[(2 * block) + 1];
    }

    const int active_pixels = std::min(srad_cfg::kBoardPixels,
                                       blocks * srad_cfg::kCudaBlockElems);
    const double pixels = static_cast<double>(active_pixels);
    const double mean = sum / pixels;
    const double variance = (sum2 / pixels) - (mean * mean);
    if (mean == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(variance / (mean * mean));
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <xclbin> [iter_cnt] [input_txt] [output_txt] [block_count]\n"
                     "       omit output_txt, or pass '-', for performance-only run\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string xclbin_path = argv[1];
    const int requested_iters =
        (argc >= 3) ? std::atoi(argv[2]) : srad_cfg::kDefaultIterations;
    const int active_iters = active_iterations(requested_iters);
    const std::string input_path =
        (argc >= 4) ? argv[3] : "./data/input_image.txt";
    const bool dump_output = argc >= 5 && std::strcmp(argv[4], "-") != 0;
    const std::string output_path = dump_output ? argv[4] : std::string();
    const int requested_blocks =
        (argc >= 6) ? std::atoi(argv[5]) : srad_cfg::kCudaBlocks;
    const int blocks = active_blocks(requested_blocks);

    std::printf("[host] image        : %d x %d\n",
                srad_cfg::kBoardRows,
                srad_cfg::kBoardCols);
    std::printf("[host] cuda block   : %d floats\n", srad_cfg::kCudaBlockElems);
    std::printf("[host] blocks       : %d / %d\n",
                blocks,
                srad_cfg::kCudaBlocks);
    std::printf("[host] requested it : %d\n", requested_iters);
    std::printf("[host] active it    : %d\n", active_iters);
    std::printf("[host] output dump  : %s\n",
                dump_output ? output_path.c_str() : "skipped");

    std::vector<float> input(srad_cfg::kBoardPixels, 0.0f);
    if (!load_input_file(input_path, input)) {
        return EXIT_FAILURE;
    }

    bool graphs_initialized = false;
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

        auto prepare = open_kernel(device, xrt_uuid, "PreparePL");
        auto reduce = open_kernel(device, xrt_uuid, "ReducePL");
        auto coeff_kernel = open_kernel(device, xrt_uuid, "CoeffPL");
        auto update = open_kernel(device, xrt_uuid, "UpdatePL");

        auto image_bo =
            xrt::bo(device, srad_cfg::kBoardImageBytes, prepare.group_id(0));
        auto output_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, update.group_id(6));
        auto sums_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, prepare.group_id(1));
        auto sums2_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, prepare.group_id(2));
        auto partials_bo =
            xrt::bo(device, blocks * srad_cfg::kReduceOutputElems * sizeof(float),
                    reduce.group_id(2));
        auto dN_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, coeff_kernel.group_id(1));
        auto dS_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, coeff_kernel.group_id(2));
        auto dW_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, coeff_kernel.group_id(3));
        auto dE_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, coeff_kernel.group_id(4));
        auto coeff_bo =
            xrt::bo(device, srad_cfg::kBoardOutputBytes, coeff_kernel.group_id(5));

        auto image_map = image_bo.map<float*>();
        auto partials_map = partials_bo.map<float*>();
        std::memcpy(image_map, input.data(), srad_cfg::kBoardImageBytes);
        image_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        graphPrepare.init();
        graphReduce.init();
        graphSradCoeff.init();
        graphSradUpdate.init();
        graphs_initialized = true;

        xrt::bo* current_bo = &image_bo;
        xrt::bo* next_bo = &output_bo;

        const auto pl_t0 = Clock::now();
        for (int iter = 0; iter < active_iters; ++iter) {
            graphPrepare.run(blocks);
            auto prepare_run =
                prepare(*current_bo, sums_bo, sums2_bo, blocks);
            prepare_run.wait();
            graphPrepare.wait();
            sums_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            sums2_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            sums_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sums2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            graphReduce.run(blocks);
            auto reduce_run = reduce(sums_bo, sums2_bo, partials_bo, blocks);
            reduce_run.wait();
            graphReduce.wait();

            partials_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float q0sqr = compute_q0sqr(partials_map, blocks);

            graphSradCoeff.run(blocks);
            auto coeff_run = coeff_kernel(*current_bo,
                                          dN_bo,
                                          dS_bo,
                                          dW_bo,
                                          dE_bo,
                                          coeff_bo,
                                          q0sqr,
                                          blocks);
            coeff_run.wait();
            graphSradCoeff.wait();
            dN_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            dS_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            dW_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            dE_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            coeff_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            dN_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            dS_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            dW_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            dE_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            coeff_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            graphSradUpdate.run(blocks);
            auto update_run = update(*current_bo,
                                     dN_bo,
                                     dS_bo,
                                     dW_bo,
                                     dE_bo,
                                     coeff_bo,
                                     *next_bo,
                                     srad_cfg::kLambdaDefault,
                                     blocks);
            update_run.wait();
            graphSradUpdate.wait();

            std::swap(current_bo, next_bo);
        }
        const auto pl_t1 = Clock::now();

        const auto pl_us =
            std::chrono::duration_cast<std::chrono::microseconds>(pl_t1 - pl_t0)
                .count();

        std::printf("pl_total_us : %lld\n", static_cast<long long>(pl_us));

        if (dump_output) {
            current_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto final_map = current_bo->map<float*>();
            dump_output_matrix(output_path, final_map);
            std::printf("[host] output       : %s\n", output_path.c_str());
        }

        graphSradUpdate.end();
        graphSradCoeff.end();
        graphReduce.end();
        graphPrepare.end();
        graphs_initialized = false;

        xrtDeviceClose(dhdl);
        dhdl = nullptr;
    } catch (const std::exception& ex) {
        if (graphs_initialized) {
            try {
                graphSradUpdate.end();
                graphSradCoeff.end();
                graphReduce.end();
                graphPrepare.end();
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
