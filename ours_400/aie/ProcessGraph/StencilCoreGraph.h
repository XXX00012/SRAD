#pragma once

#include <adf.h>
#include <string>

#include "../Config.h"
#include "../ProcessUnit/include.h"
#include "../ProcessUnit/srad.h"

using namespace adf;

// Ours SRAD AIE graph:
//   shared full halo tile with q0sqr embedded in padding -> srad_local_q ->
//       center/south/east coeff value-tag planes
//   shared full halo tile + coeff value-tags ->
//       srad_coeff_update -> J_next
//
// Each lane consumes one 19x24 halo-padded input tile from one J PLIO, then
// that lane's tile is fanned out to both compute kernels. Both kernels consume
// the full tile and emit one 16x16 output tile. q0sqr is stored in the first
// padding column of the input tile, so no separate q0 PLIO is required. PL is
// responsible for tile-major packing and boundary padding, so AIE does not use
// input margins.
//
// K2 recomputes dN/dS/dW/dE from its J input, so no gradient package is routed
// between K1 and K2 and no d_c/dN/dS/dW/dE array is exposed through PLIO/DDR.
namespace srad_plio_files {
inline std::string j_in(int lane) {
    return "./data/plio_ours_j_tile_" + std::to_string(lane) + ".txt";
}

inline std::string j_next_out(int lane) {
    return "./data/plio_ours_j_next_" + std::to_string(lane) + ".txt";
}

inline std::string j_next_out_group(int group) {
    return "./data/plio_ours_j_next_group_" + std::to_string(group) + ".txt";
}
} // namespace srad_plio_files

class SradCoreGraph : public graph {
public:
    static constexpr int kNumLanes = srad_cfg::kParallelLanes;

    port<input> in_j_k1[kNumLanes];
    port<input> in_j_k2[kNumLanes];
    port<output> out_j_next[kNumLanes];

    kernel k_local_q[kNumLanes];
    kernel k_coeff_update[kNumLanes];

    SradCoreGraph() {
#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        for (int lane = 0; lane < kNumLanes; ++lane) {
            k_local_q[lane] = kernel::create(srad_local_q);
            k_coeff_update[lane] = kernel::create(srad_coeff_update);

            source(k_local_q[lane]) = "aie/ProcessUnit/srad_local_q.cc";
            source(k_coeff_update[lane]) =
                "aie/ProcessUnit/srad_coeff_update.cc";

            headers(k_local_q[lane]) = {
                "aie/ProcessUnit/srad.h",
                "aie/ProcessUnit/include.h",
                "aie/Config.h"};
            headers(k_coeff_update[lane]) = {
                "aie/ProcessUnit/srad.h",
                "aie/ProcessUnit/include.h",
                "aie/Config.h"};

            runtime<ratio>(k_local_q[lane]) = 0.9;
            runtime<ratio>(k_coeff_update[lane]) = 0.9;
            stack_size(k_local_q[lane]) = 4096;
            stack_size(k_coeff_update[lane]) = 4096;

            location<kernel>(k_local_q[lane]) =
                tile(lane_col(lane), local_q_row(lane));
            location<kernel>(k_coeff_update[lane]) =
                tile(lane_col(lane), coeff_update_row(lane));

            auto net_j_local =
                connect<>(in_j_k1[lane], k_local_q[lane].in[0]);
            auto net_mid =
                connect<>(k_local_q[lane].out[0],
                          k_coeff_update[lane].in[0]);
            auto net_j_update =
                connect<>(in_j_k2[lane], k_coeff_update[lane].in[1]);
            auto net_out =
                connect<>(k_coeff_update[lane].out[0], out_j_next[lane]);

            dimensions(k_local_q[lane].in[0]) =
                {srad_cfg::kImageInputSampleElems};
            dimensions(k_local_q[lane].out[0]) =
                {srad_cfg::kMidElemsPerTile};
            dimensions(k_coeff_update[lane].in[0]) =
                {srad_cfg::kMidElemsPerTile};
            dimensions(k_coeff_update[lane].in[1]) =
                {srad_cfg::kImageInputSampleElems};
            dimensions(k_coeff_update[lane].out[0]) =
                {srad_cfg::kOutputSampleElems};
            dimensions(out_j_next[lane]) = {srad_cfg::kOutputSampleElems};

            fifo_depth(net_j_local) = srad_cfg::kInputObjectFifoDepth;
            fifo_depth(net_mid) = srad_cfg::kMidObjectFifoDepth;
            fifo_depth(net_j_update) =
                srad_cfg::kDelayedInputObjectFifoDepth;
            fifo_depth(net_out) = srad_cfg::kOutputObjectFifoDepth;
        }
#endif
    }

private:
    static constexpr int lane_col(int lane) {
        return lane / srad_cfg::kParallelLanesPerCol;
    }

    static constexpr int local_q_row(int lane) {
        return (lane % srad_cfg::kParallelLanesPerCol) *
               srad_cfg::kKernelsPerParallelLane;
    }

    static constexpr int coeff_update_row(int lane) {
        return local_q_row(lane) + 1;
    }
};

class GraphOursPLQ0 : public graph {
public:
    static constexpr int kNumLanes = srad_cfg::kParallelLanes;
    static constexpr int kInputLanesPerPlio = srad_cfg::kInputLanesPerPlio;
    static constexpr int kNumInputGroups = srad_cfg::kInputPlioGroups;
    static constexpr int kOutputLanesPerPlio = srad_cfg::kOutputLanesPerPlio;
    static constexpr int kNumOutputGroups = srad_cfg::kOutputPlioGroups;

    SradCoreGraph core;

    input_plio in_j[kNumInputGroups];
    adf::pktmerge<kOutputLanesPerPlio> merge_out[kNumOutputGroups];
    output_plio out_j_next[kNumOutputGroups];

    GraphOursPLQ0(const std::string& graphID) {
        for (int lane = 0; lane < kNumLanes; ++lane) {
            in_j[lane] = input_plio::create(
                graphID + "_in_j_" + std::to_string(lane),
                plio_64_bits,
                srad_plio_files::j_in(lane));

            connect<>(in_j[lane].out[0], core.in_j_k1[lane]);
            connect<>(in_j[lane].out[0], core.in_j_k2[lane]);
        }

        for (int group = 0; group < kNumOutputGroups; ++group) {
            merge_out[group] =
                adf::pktmerge<kOutputLanesPerPlio>::create();
            out_j_next[group] = output_plio::create(
                graphID + "_out_j_next_" + std::to_string(group),
                plio_64_bits,
                srad_plio_files::j_next_out_group(group));

            for (int local = 0; local < kOutputLanesPerPlio; ++local) {
                const int lane = group * kOutputLanesPerPlio + local;
                connect<window<srad_cfg::kOutputSampleBytes>, pktstream>(
                    core.out_j_next[lane], merge_out[group].in[local]);
            }

            adf::connect<adf::pktstream>(
                merge_out[group].out[0], out_j_next[group].in[0]);
        }
    }
};
