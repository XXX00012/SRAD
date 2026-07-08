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

        auto net_j_local = connect<>(in[0], k_local_q.in[0]);
        auto net_mid = connect<>(k_local_q.out[0], k_coeff_update.in[0]);
        auto net_j_update = connect<>(in[1], k_coeff_update.in[1]);
        auto net_out = connect<pktstream>(k_coeff_update.out[0], out_j_next);

        dimensions(k_local_q.in[0]) = {srad_cfg::kLocalQInputSampleElems};
        dimensions(k_local_q.out[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.in[0]) = {srad_cfg::kMidElemsPerRow};
        dimensions(k_coeff_update.in[1]) = {srad_cfg::kUpdateInputSampleElems};
        fifo_depth(net_j_local) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_mid) = srad_cfg::kMidObjectFifoDepth;
        fifo_depth(net_j_update) = srad_cfg::kDelayedInputObjectFifoDepth;
        fifo_depth(net_out) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class GraphOursPLQ0 : public graph {
public:
    static constexpr int kNumLanes = srad_cfg::kParallelLanes;
    static constexpr int kNumMergedOutputs = srad_cfg::kMergedOutputPlioCount;

    SradCoreGraph core[kNumLanes];

    input_plio in_j[kNumLanes];
    pktmerge<srad_cfg::kOutputMergeWays> out_merge[kNumMergedOutputs];
    output_plio out_j_next[kNumMergedOutputs];

    GraphOursPLQ0(const std::string& graphID) {
        for (int lane = 0; lane < kNumLanes; ++lane) {
#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
            location<kernel>(core[lane].k_local_q) =
                tile(srad_cfg::lane_tile_col(lane),
                     srad_cfg::local_q_tile_row(lane));
            location<kernel>(core[lane].k_coeff_update) =
                tile(srad_cfg::lane_tile_col(lane),
                     srad_cfg::coeff_update_tile_row(lane));
#endif

            in_j[lane] = input_plio::create(
                graphID + "_in_j_" + std::to_string(lane),
                plio_64_bits,
                "./data/plio_ours_j_" + std::to_string(lane) + ".txt");

            connect<>(in_j[lane].out[0], core[lane].in[0]);
            connect<>(in_j[lane].out[0], core[lane].in[1]);
        }

        for (int merge = 0; merge < kNumMergedOutputs; ++merge) {
            out_merge[merge] =
                pktmerge<srad_cfg::kOutputMergeWays>::create();

            out_j_next[merge] = output_plio::create(
                graphID + "_out_j_next_merged_" + std::to_string(merge),
                plio_64_bits,
                "./data/aiesim_j_next_merged_" + std::to_string(merge) + ".txt");

            for (int way = 0; way < srad_cfg::kOutputMergeWays; ++way) {
                const int lane = merge * srad_cfg::kOutputMergeWays + way;
                connect<pktstream>(core[lane].out_j_next,
                                   out_merge[merge].in[way]);
            }
            connect<pktstream>(out_merge[merge].out[0],
                               out_j_next[merge].in[0]);
        }
    }
};
