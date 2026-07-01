#pragma once

#include <adf.h>

#include <ProcessUnit/include.h>

using namespace adf;

using srad_float_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

using srad_reduce_input_buffer = input_buffer<
    float,
    adf::extents<srad_cfg::kReducePacketElems>>;

using srad_reduce_output_buffer = output_buffer<
    float,
    adf::extents<srad_cfg::kReduceOutputElems>>;

extern "C" {

void srad_prepare_kernel(srad_float_input_buffer& in_i,
                         output_buffer<float>& out_sums,
                         output_buffer<float>& out_sums2);

void srad_reduce_kernel(srad_reduce_input_buffer& in_packet,
                        srad_reduce_output_buffer& out_partial);

void srad_coeff_kernel(srad_float_input_buffer& in_neighbors,
                       srad_float_input_buffer& in_q0,
                       output_buffer<float>& out_dN,
                       output_buffer<float>& out_dS,
                       output_buffer<float>& out_dW,
                       output_buffer<float>& out_dE,
                       output_buffer<float>& out_c);

void srad_update_kernel(srad_float_input_buffer& in_update,
                        srad_float_input_buffer& in_meta,
                        output_buffer<float>& out_i_next);

}
