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
// One graph firing consumes one 19x24 halo-padded input tile from one J PLIO
// that is fanned out to both compute kernels. Both kernels consume the full
// tile and emit one 16x16 output tile. q0sqr is stored in the first padding
// column of the input tile, so no separate q0 PLIO is required. PL is
// responsible for tile-major packing and boundary padding, so AIE does not use
// input margins.
//
// K2 recomputes dN/dS/dW/dE from its J input, so no gradient package is
// routed between K1 and K2.
namespace srad_plio_files {
constexpr const char* kJIn = "./data/plio_ours_j_tile.txt";
constexpr const char* kJNextOut = "./data/plio_ours_j_next.txt";
} // namespace srad_plio_files

class SradCoreGraph : public graph {
public:
    port<input> in_j_k1;
    port<input> in_j_k2;
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
        stack_size(k_local_q) = 8192;
        stack_size(k_coeff_update) = 8192;

        location<kernel>(k_local_q) =
            tile(srad_cfg::kTileCol, srad_cfg::kLocalQTileRow);
        location<kernel>(k_coeff_update) =
            tile(srad_cfg::kTileCol, srad_cfg::kCoeffUpdateTileRow);

        auto net_j_local =
            connect<>(in_j_k1, k_local_q.in[0]);
        auto net_mid =
            connect<>(k_local_q.out[0], k_coeff_update.in[0]);
        auto net_j_update =
            connect<>(in_j_k2, k_coeff_update.in[1]);
        auto net_out =
            connect<>(k_coeff_update.out[0], out_j_next);

        dimensions(k_local_q.in[0]) = {srad_cfg::kImageInputSampleElems};
        dimensions(k_local_q.out[0]) = {srad_cfg::kMidElemsPerTile};
        dimensions(k_coeff_update.in[0]) = {srad_cfg::kMidElemsPerTile};
        dimensions(k_coeff_update.in[1]) = {srad_cfg::kImageInputSampleElems};
        dimensions(k_coeff_update.out[0]) = {srad_cfg::kOutputSampleElems};
        dimensions(out_j_next) = {srad_cfg::kOutputSampleElems};

        fifo_depth(net_j_local) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_mid) = srad_cfg::kMidObjectFifoDepth;
        fifo_depth(net_j_update) = srad_cfg::kDelayedInputObjectFifoDepth;
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
        in_j = input_plio::create(graphID + "_in_j",
                                  plio_64_bits,
                                  srad_plio_files::kJIn);
        out_j_next = output_plio::create(graphID + "_out_j_next",
                                         plio_64_bits,
                                         srad_plio_files::kJNextOut);

        connect<>(in_j.out[0], core.in_j_k1);
        connect<>(in_j.out[0], core.in_j_k2);
        connect<>(core.out_j_next, out_j_next.in[0]);
    }
};
