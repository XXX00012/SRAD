#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

// Single-kernel FPGA-v5-style graph.
// The ProcessGraph layer is intentionally thin, but keeps this baseline aligned
// with the staged SRAD/CUDA and SRAD/CUDA_pl project structure.
namespace srad_plio_files {
constexpr const char* kReduceIn = "./data/plio_fpga_reduce.txt";
constexpr const char* kComputeIn = "./data/plio_fpga_compute.txt";
constexpr const char* kLambdaIn = "./data/plio_fpga_lambda.txt";
constexpr const char* kJNextOut = "./data/plio_fpga_j_next.txt";
} // namespace srad_plio_files

class GraphFpgaV5 : public graph {
public:
    input_plio in_reduce;
    input_plio in_compute;
    input_plio in_lambda;
    output_plio out_j_next;
    kernel k_fused;

    GraphFpgaV5(const std::string& graphID) {
        in_reduce = input_plio::create(graphID + "_in_reduce",
                                       plio_32_bits,
                                       srad_plio_files::kReduceIn);
        in_compute = input_plio::create(graphID + "_in_compute",
                                        plio_32_bits,
                                        srad_plio_files::kComputeIn);
        in_lambda = input_plio::create(graphID + "_in_lambda",
                                       plio_32_bits,
                                       srad_plio_files::kLambdaIn);
        out_j_next = output_plio::create(graphID + "_out_j_next",
                                         plio_32_bits,
                                         srad_plio_files::kJNextOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_fused = kernel::create(srad_fpga_v5_fused);
        source(k_fused) = "aie/ProcessUnit/srad_fpga.cc";
        headers(k_fused) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_fused) = 0.9;
        location<kernel>(k_fused) = tile(srad_cfg::kTileCol, srad_cfg::kFusedTileRow);

        auto net_reduce = connect<>(in_reduce.out[0], k_fused.in[0]);
        auto net_compute = connect<>(in_compute.out[0], k_fused.in[1]);
        auto net_lambda = connect<>(in_lambda.out[0], k_fused.in[2]);
        auto net_out = connect<>(k_fused.out[0], out_j_next.in[0]);

        dimensions(k_fused.in[0]) = {srad_cfg::kPixels};
        dimensions(k_fused.in[1]) = {srad_cfg::kPixels};
        dimensions(k_fused.in[2]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_fused.out[0]) = {srad_cfg::kPixels};

        fifo_depth(net_reduce) = 2;
        fifo_depth(net_compute) = 2;
        fifo_depth(net_lambda) = 2;
        fifo_depth(net_out) = 2;
#endif
    }
};
