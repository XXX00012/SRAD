#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

namespace srad_plio_files {
constexpr const char* kParamsIn = "./data/plio_fpga_params.txt";
constexpr const char* kComputeIn = "./data/plio_fpga_compute.txt";
constexpr const char* kIndexOut = "./data/plio_fpga_j_next_index.txt";
constexpr const char* kValueOut = "./data/plio_fpga_j_next_value.txt";
} // namespace srad_plio_files

class GraphFpgaV5 : public graph {
public:
    input_plio in_params;
    input_plio in_compute;
    output_plio out_index;
    output_plio out_value;
    kernel k_srad;

    GraphFpgaV5(const std::string& graphID) {
        in_params = input_plio::create(graphID + "_in_params",
                                       plio_32_bits,
                                       srad_plio_files::kParamsIn);
        in_compute = input_plio::create(graphID + "_in_compute",
                                        plio_32_bits,
                                        srad_plio_files::kComputeIn);
        out_index = output_plio::create(graphID + "_out_index",
                                        plio_32_bits,
                                        srad_plio_files::kIndexOut);
        out_value = output_plio::create(graphID + "_out_value",
                                        plio_32_bits,
                                        srad_plio_files::kValueOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_srad = kernel::create(srad_fpga_v5);
        source(k_srad) = "aie/ProcessUnit/srad_fpga.cc";
        headers(k_srad) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_srad) = 0.9;
        stack_size(k_srad) = 8192;
        location<kernel>(k_srad) =
            tile(srad_cfg::kTileCol, srad_cfg::kFusedTileRow);

        auto net_params = connect<>(in_params.out[0], k_srad.in[0]);
        auto net_compute = connect<>(in_compute.out[0], k_srad.in[1]);
        auto net_index = connect<>(k_srad.out[0], out_index.in[0]);
        auto net_value = connect<>(k_srad.out[1], out_value.in[0]);

        fifo_depth(net_params) = 2;
        fifo_depth(net_compute) = 2;
        fifo_depth(net_index) = 2;
        fifo_depth(net_value) = 2;
#endif
    }
};
