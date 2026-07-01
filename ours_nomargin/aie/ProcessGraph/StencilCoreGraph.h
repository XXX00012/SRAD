#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

class SradCoreGraph : public graph {
public:
    static constexpr int kInputCount = 1;

    port<input> in[kInputCount];
    port<output> out_j_next;

    kernel k_local_q;
    kernel k_coeff_update;

    SradCoreGraph() {
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
        stack_size(k_local_q) = 4096;
        stack_size(k_coeff_update) = 4096;

        location<kernel>(k_local_q) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kLocalQTileRow);
        location<kernel>(k_coeff_update) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kCoeffUpdateTileRow);

        auto net_j_local = connect<>(in[0], k_local_q.in[0]);
        auto net_mid = connect<>(k_local_q.out[0], k_coeff_update.in[0]);
        auto net_out = connect<>(k_coeff_update.out[0], out_j_next);

        dimensions(k_local_q.in[0]) = {srad_cfg::kLocalQInputSampleElems};
        dimensions(k_local_q.out[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.in[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.out[0]) = {srad_cfg::kOutputRowPhysElems};
        dimensions(out_j_next) = {srad_cfg::kOutputRowPhysElems};

        fifo_depth(net_j_local) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_mid) = srad_cfg::kMidObjectFifoDepth;
        fifo_depth(net_out) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class GraphOursPLQ0 : public graph {
public:
    SradCoreGraph core;

    input_plio in_j;
    output_plio out_j_next;

    GraphOursPLQ0(const std::string& graphID) {
        in_j = input_plio::create(
            graphID + "_in_j_0",
            plio_64_bits,
            "./data/plio_ours_j.txt");

        out_j_next = output_plio::create(
            graphID + "_out_j_next_0",
            plio_64_bits,
            "./data/aiesim_j_next.txt");

        connect<>(in_j.out[0], core.in[0]);
        connect<>(core.out_j_next, out_j_next.in[0]);
    }
};
