#pragma once

#include <adf.h>
#include "Config.h"

using namespace adf;

using srad_image_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

using srad_scalar_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

extern "C" {

void srad_prepare_kernel(
    srad_image_input_buffer& in_j,
    output_buffer<float>& out_sums,
    output_buffer<float>& out_sums2);

void srad_reduce_kernel(
    srad_image_input_buffer& in_sums,
    srad_image_input_buffer& in_sums2,
    output_buffer<float>& out_stats);

void srad_kernel(
    srad_image_input_buffer& in_j,
    srad_scalar_input_buffer& in_q0sqr,
    output_buffer<float>& out_c,
    output_buffer<float>& out_dN,
    output_buffer<float>& out_dS,
    output_buffer<float>& out_dW,
    output_buffer<float>& out_dE);

void srad2_kernel(
    srad_image_input_buffer& in_j,
    srad_image_input_buffer& in_c,
    srad_image_input_buffer& in_dN,
    srad_image_input_buffer& in_dS,
    srad_image_input_buffer& in_dW,
    srad_image_input_buffer& in_dE,
    srad_scalar_input_buffer& in_lambda,
    output_buffer<float>& out_j_next);

}
