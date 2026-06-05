#pragma once

#include <adf.h>
#include <string>

#include "Config.h"
#include "ProcessUnit/include.h"
#include "ProcessUnit/srad.h"

using namespace adf;

namespace srad_plio_files {
constexpr const char* kPrepareJ = "./data/plio_prepare_j.txt";
constexpr const char* kPrepareSumsOut = "./data/plio_prepare_sums.txt";
constexpr const char* kPrepareSums2Out = "./data/plio_prepare_sums2.txt";
constexpr const char* kReduceSums = kPrepareSumsOut;
constexpr const char* kReduceSums2 = kPrepareSums2Out;
constexpr const char* kCoeffJ = "./data/plio_coeff_j.txt";
constexpr const char* kCoeffQ0 = "./data/plio_runtime_coeff_q0sqr.txt";
constexpr const char* kUpdateJ = "./data/plio_update_j.txt";
constexpr const char* kUpdateC = "./data/plio_runtime_update_c.txt";
constexpr const char* kUpdateDN = "./data/plio_runtime_update_dN.txt";
constexpr const char* kUpdateDS = "./data/plio_runtime_update_dS.txt";
constexpr const char* kUpdateDW = "./data/plio_runtime_update_dW.txt";
constexpr const char* kUpdateDE = "./data/plio_runtime_update_dE.txt";
constexpr const char* kUpdateLambda = "./data/plio_runtime_lambda.txt";
constexpr const char* kReduceStatsOut = "./data/plio_reduce_stats.txt";
constexpr const char* kCoeffCOut = "./data/plio_coeff_c.txt";
constexpr const char* kCoeffDNOut = "./data/plio_coeff_dN.txt";
constexpr const char* kCoeffDSOut = "./data/plio_coeff_dS.txt";
constexpr const char* kCoeffDWOut = "./data/plio_coeff_dW.txt";
constexpr const char* kCoeffDEOut = "./data/plio_coeff_dE.txt";
constexpr const char* kUpdateOut = "./data/plio_update_j_next.txt";
} // namespace srad_plio_files

class GraphPrepare : public graph {
public:
    input_plio in_j;
    output_plio out_sums;
    output_plio out_sums2;
    kernel k_prepare;

    GraphPrepare(const std::string& graphID) {
        in_j = input_plio::create(graphID + "_in_j",
                                  plio_32_bits,
                                  srad_plio_files::kPrepareJ);
        out_sums = output_plio::create(graphID + "_out_sums",
                                       plio_32_bits,
                                       srad_plio_files::kPrepareSumsOut);
        out_sums2 = output_plio::create(graphID + "_out_sums2",
                                        plio_32_bits,
                                        srad_plio_files::kPrepareSums2Out);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_prepare = kernel::create(srad_prepare_kernel);
        source(k_prepare) = "aie/ProcessUnit/srad_prepare.cc";
        headers(k_prepare) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_prepare) = 0.9;
        location<kernel>(k_prepare) = tile(srad_cfg::kTileCol, srad_cfg::kPrepareTileRow);

        auto net_in = connect<>(in_j.out[0], k_prepare.in[0]);
        auto net_sums = connect<>(k_prepare.out[0], out_sums.in[0]);
        auto net_sums2 = connect<>(k_prepare.out[1], out_sums2.in[0]);

        dimensions(k_prepare.in[0]) = {srad_cfg::kPixels};
        dimensions(k_prepare.out[0]) = {srad_cfg::kPixels};
        dimensions(k_prepare.out[1]) = {srad_cfg::kPixels};

        fifo_depth(net_in) = 2;
        fifo_depth(net_sums) = 2;
        fifo_depth(net_sums2) = 2;
#endif
    }
};

class GraphReduce : public graph {
public:
    input_plio in_sums;
    input_plio in_sums2;
    output_plio out_stats;
    kernel k_reduce;

    GraphReduce(const std::string& graphID) {
        in_sums = input_plio::create(graphID + "_in_sums",
                                     plio_32_bits,
                                     srad_plio_files::kReduceSums);
        in_sums2 = input_plio::create(graphID + "_in_sums2",
                                      plio_32_bits,
                                      srad_plio_files::kReduceSums2);
        out_stats = output_plio::create(graphID + "_out_stats",
                                        plio_32_bits,
                                        srad_plio_files::kReduceStatsOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_reduce = kernel::create(srad_reduce_kernel);
        source(k_reduce) = "aie/ProcessUnit/srad_reduce_q0.cc";
        headers(k_reduce) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_reduce) = 0.9;
        stack_size(k_reduce) = 16384;
        location<kernel>(k_reduce) = tile(srad_cfg::kTileCol, srad_cfg::kReduceTileRow);

        auto net_sums = connect<>(in_sums.out[0], k_reduce.in[0]);
        auto net_sums2 = connect<>(in_sums2.out[0], k_reduce.in[1]);
        auto net_out = connect<>(k_reduce.out[0], out_stats.in[0]);

        dimensions(k_reduce.in[0]) = {srad_cfg::kPixels};
        dimensions(k_reduce.in[1]) = {srad_cfg::kPixels};
        dimensions(k_reduce.out[0]) = {srad_cfg::kScalarPacketElems};

        fifo_depth(net_sums) = 2;
        fifo_depth(net_sums2) = 2;
        fifo_depth(net_out) = 2;
#endif
    }
};

class GraphCoeff : public graph {
public:
    input_plio in_j;
    input_plio in_q0sqr;
    output_plio out_c;
    output_plio out_dN;
    output_plio out_dS;
    output_plio out_dW;
    output_plio out_dE;
    kernel k_coeff;

    GraphCoeff(const std::string& graphID) {
        in_j = input_plio::create(graphID + "_in_j",
                                  plio_32_bits,
                                  srad_plio_files::kCoeffJ);
        in_q0sqr = input_plio::create(graphID + "_in_q0sqr",
                                      plio_32_bits,
                                      srad_plio_files::kCoeffQ0);
        out_c = output_plio::create(graphID + "_out_c",
                                    plio_32_bits,
                                    srad_plio_files::kCoeffCOut);
        out_dN = output_plio::create(graphID + "_out_dN",
                                     plio_32_bits,
                                     srad_plio_files::kCoeffDNOut);
        out_dS = output_plio::create(graphID + "_out_dS",
                                     plio_32_bits,
                                     srad_plio_files::kCoeffDSOut);
        out_dW = output_plio::create(graphID + "_out_dW",
                                     plio_32_bits,
                                     srad_plio_files::kCoeffDWOut);
        out_dE = output_plio::create(graphID + "_out_dE",
                                     plio_32_bits,
                                     srad_plio_files::kCoeffDEOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_coeff = kernel::create(srad_kernel);
        source(k_coeff) = "aie/ProcessUnit/srad_coeff.cc";
        headers(k_coeff) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_coeff) = 0.9;
        location<kernel>(k_coeff) = tile(srad_cfg::kTileCol, srad_cfg::kCoeffTileRow);

        auto net_j = connect<>(in_j.out[0], k_coeff.in[0]);
        auto net_q0 = connect<>(in_q0sqr.out[0], k_coeff.in[1]);
        auto net_c = connect<>(k_coeff.out[0], out_c.in[0]);
        auto net_dN = connect<>(k_coeff.out[1], out_dN.in[0]);
        auto net_dS = connect<>(k_coeff.out[2], out_dS.in[0]);
        auto net_dW = connect<>(k_coeff.out[3], out_dW.in[0]);
        auto net_dE = connect<>(k_coeff.out[4], out_dE.in[0]);

        dimensions(k_coeff.in[0]) = {srad_cfg::kPixels};
        dimensions(k_coeff.in[1]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_coeff.out[0]) = {srad_cfg::kBlockPixels};
        dimensions(k_coeff.out[1]) = {srad_cfg::kBlockPixels};
        dimensions(k_coeff.out[2]) = {srad_cfg::kBlockPixels};
        dimensions(k_coeff.out[3]) = {srad_cfg::kBlockPixels};
        dimensions(k_coeff.out[4]) = {srad_cfg::kBlockPixels};

        fifo_depth(net_j) = 2;
        fifo_depth(net_q0) = 2;
        fifo_depth(net_c) = 2;
        fifo_depth(net_dN) = 2;
        fifo_depth(net_dS) = 2;
        fifo_depth(net_dW) = 2;
        fifo_depth(net_dE) = 2;
#endif
    }
};

class GraphUpdate : public graph {
public:
    input_plio in_j;
    input_plio in_c;
    input_plio in_dN;
    input_plio in_dS;
    input_plio in_dW;
    input_plio in_dE;
    input_plio in_lambda;
    output_plio out_j_next;
    kernel k_update;

    GraphUpdate(const std::string& graphID) {
        in_j = input_plio::create(graphID + "_in_j",
                                  plio_32_bits,
                                  srad_plio_files::kUpdateJ);
        in_c = input_plio::create(graphID + "_in_c",
                                  plio_32_bits,
                                  srad_plio_files::kUpdateC);
        in_dN = input_plio::create(graphID + "_in_dN",
                                   plio_32_bits,
                                   srad_plio_files::kUpdateDN);
        in_dS = input_plio::create(graphID + "_in_dS",
                                   plio_32_bits,
                                   srad_plio_files::kUpdateDS);
        in_dW = input_plio::create(graphID + "_in_dW",
                                   plio_32_bits,
                                   srad_plio_files::kUpdateDW);
        in_dE = input_plio::create(graphID + "_in_dE",
                                   plio_32_bits,
                                   srad_plio_files::kUpdateDE);
        in_lambda = input_plio::create(graphID + "_in_lambda",
                                       plio_32_bits,
                                       srad_plio_files::kUpdateLambda);
        out_j_next = output_plio::create(graphID + "_out_j_next",
                                         plio_32_bits,
                                         srad_plio_files::kUpdateOut);

#if defined(__AIESIM__) || defined(__X86SIM__) || defined(__ADF_FRONTEND__)
        k_update = kernel::create(srad2_kernel);
        source(k_update) = "aie/ProcessUnit/srad_update.cc";
        headers(k_update) = {
            "aie/ProcessUnit/srad.h",
            "aie/ProcessUnit/include.h",
            "aie/Config.h"};
        runtime<ratio>(k_update) = 0.9;
        location<kernel>(k_update) = tile(srad_cfg::kTileCol, srad_cfg::kUpdateTileRow);

        auto net_j = connect<>(in_j.out[0], k_update.in[0]);
        auto net_c = connect<>(in_c.out[0], k_update.in[1]);
        auto net_dN = connect<>(in_dN.out[0], k_update.in[2]);
        auto net_dS = connect<>(in_dS.out[0], k_update.in[3]);
        auto net_dW = connect<>(in_dW.out[0], k_update.in[4]);
        auto net_dE = connect<>(in_dE.out[0], k_update.in[5]);
        auto net_lambda = connect<>(in_lambda.out[0], k_update.in[6]);
        auto net_out = connect<>(k_update.out[0], out_j_next.in[0]);

        dimensions(k_update.in[0]) = {srad_cfg::kBlockPixels};
        dimensions(k_update.in[1]) = {srad_cfg::kUpdateCInputElems};
        dimensions(k_update.in[2]) = {srad_cfg::kBlockPixels};
        dimensions(k_update.in[3]) = {srad_cfg::kBlockPixels};
        dimensions(k_update.in[4]) = {srad_cfg::kBlockPixels};
        dimensions(k_update.in[5]) = {srad_cfg::kBlockPixels};
        dimensions(k_update.in[6]) = {srad_cfg::kScalarPacketElems};
        dimensions(k_update.out[0]) = {srad_cfg::kBlockPixels};

        fifo_depth(net_j) = 2;
        fifo_depth(net_c) = 2;
        fifo_depth(net_dN) = 2;
        fifo_depth(net_dS) = 2;
        fifo_depth(net_dW) = 2;
        fifo_depth(net_dE) = 2;
        fifo_depth(net_lambda) = 2;
        fifo_depth(net_out) = 2;
#endif
    }
};

extern GraphPrepare graphPrepare;
extern GraphReduce graphReduce;
extern GraphCoeff graphCoeff;
extern GraphUpdate graphUpdate;

#if (defined(__AIESIM__) || defined(__X86SIM__)) && defined(__PS_INIT_AIE__)
int ps_main();
#endif
