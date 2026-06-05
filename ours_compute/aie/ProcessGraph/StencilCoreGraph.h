#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

// Ours-compute SRAD AIE graph:
//   J -> srad_local_q -> AIE local mid buffer -> srad_coeff_update -> J_next
//                                      q0sqr GMIO ------------^
//                                      lambda from GMIO ------^
//
// One graph firing consumes one packed 8x8 block. A 16x16 image is sent as
// four graph firings.
//
// The K1/K2 boundary is an AIE-local object FIFO buffer. Each pixel record
// carries six float32 values in order: unused, dN, dS, dW, dE, JC.
//
// No d_c/dN/dS/dW/dE array is exposed through GMIO/DDR.
class GraphOursPLQ0 : public graph {
public:
    input_gmio in_j;
    input_gmio in_q0sqr;
    input_gmio in_lambda;
    output_gmio out_j_next;

    kernel k_local_q;
    kernel k_coeff_update;

    GraphOursPLQ0(const std::string& graphID) {
        in_j = input_gmio::create(graphID + "_in_j", 64, 1000);
        in_q0sqr = input_gmio::create(graphID + "_in_q0sqr", 64, 1000);
        in_lambda = input_gmio::create(graphID + "_in_lambda", 64, 1000);
        out_j_next = output_gmio::create(graphID + "_out_j_next", 64, 1000);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_local_q = kernel::create(srad_local_q);
        k_coeff_update = kernel::create(srad_coeff_update);

        source(k_local_q) = "aie/ProcessUnit/srad_local_q.cc";
        source(k_coeff_update) = "aie/ProcessUnit/srad_coeff_update.cc";

        headers(k_local_q) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        headers(k_coeff_update) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};

        runtime<ratio>(k_local_q) = 0.9;
        runtime<ratio>(k_coeff_update) = 0.9;

        location<kernel>(k_local_q) =
            tile(srad_cfg::kTileCol, srad_cfg::kLocalQTileRow);
        location<kernel>(k_coeff_update) =
            tile(srad_cfg::kTileCol, srad_cfg::kCoeffUpdateTileRow);

        auto net_j = connect<>(in_j.out[0], k_local_q.in[0]);
        auto net_mid =
            connect<>(k_local_q.out[0], k_coeff_update.in[0]);
        auto net_q0sqr =
            connect<>(in_q0sqr.out[0], k_coeff_update.in[1]);
        auto net_lambda =
            connect<>(in_lambda.out[0], k_coeff_update.in[2]);
        auto net_out =
            connect<>(k_coeff_update.out[0], out_j_next.in[0]);

        dimensions(k_local_q.in[0]) = {srad_cfg::kBlockPixels};
        dimensions(k_local_q.out[0]) = {srad_cfg::kMidElemsPerBlock};
        dimensions(k_coeff_update.in[0]) = {srad_cfg::kMidElemsPerBlock};
        dimensions(k_coeff_update.in[1]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_coeff_update.in[2]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_coeff_update.out[0]) = {srad_cfg::kBlockPixels};

        fifo_depth(net_j) = 2;
        fifo_depth(net_mid) = 2;
        fifo_depth(net_q0sqr) = 2;
        fifo_depth(net_lambda) = 2;
        fifo_depth(net_out) = 2;
#endif
    }
};
