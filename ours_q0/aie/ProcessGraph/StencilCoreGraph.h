#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

class SradCoreGraph : public graph {
public:
    static constexpr int kInputCount = 2;

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
        auto net_j_update = connect<>(in[1], k_coeff_update.in[1]);
        auto net_out = connect<>(k_coeff_update.out[0], out_j_next);

        dimensions(k_local_q.in[0]) = {srad_cfg::kLocalQInputSampleElems};
        dimensions(k_local_q.out[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.in[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.in[1]) = {srad_cfg::kUpdateInputSampleElems};
        dimensions(k_coeff_update.out[0]) = {srad_cfg::kOutputRowPhysElems};
        dimensions(out_j_next) = {srad_cfg::kOutputRowPhysElems};

        fifo_depth(net_j_local) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_mid) = srad_cfg::kMidObjectFifoDepth;
        fifo_depth(net_j_update) = srad_cfg::kDelayedInputObjectFifoDepth;
        fifo_depth(net_out) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class SradStatsGraph : public graph {
public:
    port<input> in_j;
    port<output> out_stats;

    kernel k_row_stats;

    SradStatsGraph() {
#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_row_stats = kernel::create(srad_row_stats);
        source(k_row_stats) = "aie/ProcessUnit/srad_row_stats.cc";
        headers(k_row_stats) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};

        runtime<ratio>(k_row_stats) = 0.9;
        stack_size(k_row_stats) = 2048;
        location<kernel>(k_row_stats) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kStatsTileRow);

        auto net_in = connect<>(in_j, k_row_stats.in[0]);
        auto net_out = connect<>(k_row_stats.out[0], out_stats);

        dimensions(k_row_stats.in[0]) = {srad_cfg::kStatsInputSampleElems};
        dimensions(k_row_stats.out[0]) = {srad_cfg::kStatsOutputElems};
        dimensions(out_stats) = {srad_cfg::kStatsOutputElems};

        fifo_depth(net_in) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_out) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class GraphOursPLQ0 : public graph {
public:
    SradStatsGraph stats;
    SradCoreGraph core;

    input_plio in_j_stats;
    input_plio in_j_update;
    output_plio out_stats;
    output_plio out_j_next;

    GraphOursPLQ0(const std::string& graphID) {
        in_j_stats = input_plio::create(
            graphID + "_in_j_stats_0",
            plio_64_bits,
            "./data/plio_ours_j_stats.txt");

        in_j_update = input_plio::create(
            graphID + "_in_j_update_0",
            plio_64_bits,
            "./data/plio_ours_j.txt");

        out_stats = output_plio::create(
            graphID + "_out_stats_0",
            plio_64_bits,
            "./data/aiesim_row_stats.txt");

        out_j_next = output_plio::create(
            graphID + "_out_j_next_0",
            plio_64_bits,
            "./data/aiesim_j_next.txt");

        connect<>(in_j_stats.out[0], stats.in_j);
        connect<>(stats.out_stats, out_stats.in[0]);
        connect<>(in_j_update.out[0], core.in[0]);
        connect<>(in_j_update.out[0], core.in[1]);
        connect<>(core.out_j_next, out_j_next.in[0]);
    }
};
