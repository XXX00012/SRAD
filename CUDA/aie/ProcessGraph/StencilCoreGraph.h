#pragma once

#include <adf.h>
#include <string>

#include <Config.h>
#include <ProcessUnit/include.h>
#include <ProcessUnit/srad.h>

using namespace adf;

class GraphPrepare : public graph {
public:
    input_plio in_i;
    output_plio out_sums;
    output_plio out_sums2;
    kernel k_prepare;

    GraphPrepare(const std::string& graphID) {
        in_i = input_plio::create(
            graphID + "_in_i",
            plio_64_bits,
            "./data/gpu_prepare_i.txt");
        out_sums = output_plio::create(
            graphID + "_out_sums",
            plio_64_bits,
            "./data/gpu_prepare_sums.txt");
        out_sums2 = output_plio::create(
            graphID + "_out_sums2",
            plio_64_bits,
            "./data/gpu_prepare_sums2.txt");

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_prepare = kernel::create(srad_prepare_kernel);
        source(k_prepare) = "aie/ProcessUnit/srad_prepare.cc";
        headers(k_prepare) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_prepare) = 0.9;
        stack_size(k_prepare) = 2048;
        location<kernel>(k_prepare) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kPrepareTileRow);

        auto net_in = connect<>(in_i.out[0], k_prepare.in[0]);
        auto net_sum = connect<>(k_prepare.out[0], out_sums.in[0]);
        auto net_sum2 = connect<>(k_prepare.out[1], out_sums2.in[0]);

        dimensions(k_prepare.in[0]) = {srad_cfg::kPrepareInputElems};
        dimensions(k_prepare.out[0]) = {srad_cfg::kPrepareOutputElems};
        dimensions(k_prepare.out[1]) = {srad_cfg::kPrepareOutputElems};
        fifo_depth(net_in) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_sum) = srad_cfg::kOutputObjectFifoDepth;
        fifo_depth(net_sum2) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class GraphReduce : public graph {
public:
    input_plio in_packet;
    output_plio out_partial;
    kernel k_reduce;

    GraphReduce(const std::string& graphID) {
        in_packet = input_plio::create(
            graphID + "_in_packet",
            plio_64_bits,
            "./data/gpu_reduce_packet.txt");
        out_partial = output_plio::create(
            graphID + "_out_partial",
            plio_64_bits,
            "./data/gpu_reduce_partial.txt");

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_reduce = kernel::create(srad_reduce_kernel);
        source(k_reduce) = "aie/ProcessUnit/srad_reduce.cc";
        headers(k_reduce) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_reduce) = 0.9;
        stack_size(k_reduce) = 8192;
        location<kernel>(k_reduce) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kReduceTileRow);

        auto net_packet = connect<>(in_packet.out[0], k_reduce.in[0]);
        auto net_partial = connect<>(k_reduce.out[0], out_partial.in[0]);

        dimensions(k_reduce.in[0]) = {srad_cfg::kReducePacketElems};
        dimensions(k_reduce.out[0]) = {srad_cfg::kReduceOutputElems};
        fifo_depth(net_packet) = srad_cfg::kInputObjectFifoDepth;
        fifo_depth(net_partial) = srad_cfg::kOutputObjectFifoDepth;
#endif
    }
};

class GraphSradCoeff : public graph {
public:
    input_plio in_neighbors;
    input_plio in_q0;
    output_plio out_dN;
    output_plio out_dS;
    output_plio out_dW;
    output_plio out_dE;
    output_plio out_c;
    kernel k_coeff;

    GraphSradCoeff(const std::string& graphID) {
        in_neighbors = input_plio::create(
            graphID + "_in_neighbors",
            plio_64_bits,
            "./data/gpu_srad_neighbors.txt");
        in_q0 = input_plio::create(
            graphID + "_in_q0",
            plio_64_bits,
            "./data/gpu_srad_q0.txt");
        out_dN = output_plio::create(
            graphID + "_out_dN",
            plio_64_bits,
            "./data/gpu_srad_dN.txt");
        out_dS = output_plio::create(
            graphID + "_out_dS",
            plio_64_bits,
            "./data/gpu_srad_dS.txt");
        out_dW = output_plio::create(
            graphID + "_out_dW",
            plio_64_bits,
            "./data/gpu_srad_dW.txt");
        out_dE = output_plio::create(
            graphID + "_out_dE",
            plio_64_bits,
            "./data/gpu_srad_dE.txt");
        out_c = output_plio::create(
            graphID + "_out_c",
            plio_64_bits,
            "./data/gpu_srad_c.txt");

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_coeff = kernel::create(srad_coeff_kernel);
        source(k_coeff) = "aie/ProcessUnit/srad_coeff.cc";
        headers(k_coeff) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_coeff) = 0.9;
        stack_size(k_coeff) = 4096;
        location<kernel>(k_coeff) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kCoeffTileRow);

        connect<>(in_neighbors.out[0], k_coeff.in[0]);
        connect<>(in_q0.out[0], k_coeff.in[1]);
        connect<>(k_coeff.out[0], out_dN.in[0]);
        connect<>(k_coeff.out[1], out_dS.in[0]);
        connect<>(k_coeff.out[2], out_dW.in[0]);
        connect<>(k_coeff.out[3], out_dE.in[0]);
        connect<>(k_coeff.out[4], out_c.in[0]);

        dimensions(k_coeff.in[0]) = {srad_cfg::kCoeffInputElems};
        dimensions(k_coeff.in[1]) = {srad_cfg::kMetaPacketElems};
        dimensions(k_coeff.out[0]) = {srad_cfg::kCoeffOutputElems};
        dimensions(k_coeff.out[1]) = {srad_cfg::kCoeffOutputElems};
        dimensions(k_coeff.out[2]) = {srad_cfg::kCoeffOutputElems};
        dimensions(k_coeff.out[3]) = {srad_cfg::kCoeffOutputElems};
        dimensions(k_coeff.out[4]) = {srad_cfg::kCoeffOutputElems};
#endif
    }
};

class GraphSradUpdate : public graph {
public:
    input_plio in_update;
    input_plio in_meta;
    output_plio out_i_next;
    kernel k_update;

    GraphSradUpdate(const std::string& graphID) {
        in_update = input_plio::create(
            graphID + "_in_update",
            plio_64_bits,
            "./data/gpu_srad2_update.txt");
        in_meta = input_plio::create(
            graphID + "_in_meta",
            plio_64_bits,
            "./data/gpu_srad2_meta.txt");
        out_i_next = output_plio::create(
            graphID + "_out_i_next",
            plio_64_bits,
            "./data/gpu_srad2_i_next.txt");

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_update = kernel::create(srad_update_kernel);
        source(k_update) = "aie/ProcessUnit/srad_update.cc";
        headers(k_update) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_update) = 0.9;
        stack_size(k_update) = 4096;
        location<kernel>(k_update) =
            tile(srad_cfg::kAieTileCol, srad_cfg::kUpdateTileRow);

        connect<>(in_update.out[0], k_update.in[0]);
        connect<>(in_meta.out[0], k_update.in[1]);
        connect<>(k_update.out[0], out_i_next.in[0]);

        dimensions(k_update.in[0]) = {srad_cfg::kUpdateInputElems};
        dimensions(k_update.in[1]) = {srad_cfg::kMetaPacketElems};
        dimensions(k_update.out[0]) = {srad_cfg::kUpdateOutputElems};
#endif
    }
};
