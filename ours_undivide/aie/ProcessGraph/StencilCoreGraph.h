#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

// Ours-undivide SRAD AIE graph:
//   J coeff  -> srad_undivide_fused -> J_next
//   J update ----------------^
//   q0sqr PLIO -------------^
//   lambda PLIO ------------^
//
// Each graph firing consumes one packed 16x16 tile from a 16x16 image. The
// fused kernel processes one row at a time, computes coefficients and
// gradients directly, then updates that row. The per-tile intermediates
// stay local to the AIE core instead of materializing the whole mid plane
// through PLIO/DDR.
//
// No d_c/dN/dS/dW/dE array is exposed through PLIO/DDR.
namespace srad_plio_files {
constexpr const char* kJIn = "./data/plio_ours_undivide_j.txt";
constexpr const char* kJUpdateIn = "./data/plio_ours_undivide_j_update.txt";
constexpr const char* kQ0In = "./data/plio_ours_undivide_q0sqr.txt";
constexpr const char* kLambdaIn = "./data/plio_ours_undivide_lambda.txt";
constexpr const char* kJNextOut = "./data/plio_ours_undivide_j_next.txt";
} // namespace srad_plio_files

class GraphOursPLQ0 : public graph {
public:
    input_plio in_j;
    input_plio in_j_update;
    input_plio in_q0sqr;
    input_plio in_lambda;
    output_plio out_j_next;

    kernel k_fused;

    GraphOursPLQ0(const std::string& graphID) {
        in_j = input_plio::create(graphID + "_in_j",
                                  plio_64_bits,
                                  srad_plio_files::kJIn);
        in_j_update = input_plio::create(graphID + "_in_j_update",
                                         plio_64_bits,
                                         srad_plio_files::kJUpdateIn);
        in_q0sqr = input_plio::create(graphID + "_in_q0sqr",
                                      plio_64_bits,
                                      srad_plio_files::kQ0In);
        in_lambda = input_plio::create(graphID + "_in_lambda",
                                       plio_64_bits,
                                       srad_plio_files::kLambdaIn);
        out_j_next = output_plio::create(graphID + "_out_j_next",
                                         plio_64_bits,
                                         srad_plio_files::kJNextOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_fused = kernel::create(srad_undivide_fused);

        source(k_fused) = "aie/ProcessUnit/srad_undivide_fused.cc";

        headers(k_fused) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};

        runtime<ratio>(k_fused) = 0.9;
        stack_size(k_fused) = 16384;

        location<kernel>(k_fused) =
            tile(srad_cfg::kTileCol, srad_cfg::kLocalQTileRow);

        auto net_j = connect<>(in_j.out[0], k_fused.in[0]);
        auto net_j_update = connect<>(in_j_update.out[0], k_fused.in[1]);
        auto net_q0sqr =
            connect<>(in_q0sqr.out[0], k_fused.in[2]);
        auto net_lambda =
            connect<>(in_lambda.out[0], k_fused.in[3]);
        auto net_out =
            connect<>(k_fused.out[0], out_j_next.in[0]);

        dimensions(k_fused.in[0]) = {srad_cfg::kBlockPixels};
        dimensions(k_fused.in[1]) = {srad_cfg::kBlockPixels};
        dimensions(k_fused.in[2]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_fused.in[3]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_fused.out[0]) = {srad_cfg::kBlockPixels};

        fifo_depth(net_j) = 2;
        fifo_depth(net_j_update) = 2;
        fifo_depth(net_q0sqr) = 2;
        fifo_depth(net_lambda) = 2;
        fifo_depth(net_out) = 2;
#endif
    }
};
