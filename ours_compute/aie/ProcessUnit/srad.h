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

using srad_mid_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

extern "C" {

void srad_local_q(
    srad_image_input_buffer& in_j,
    output_buffer<float>& mid_buffer);

void srad_coeff_update(
    srad_mid_input_buffer& mid_buffer,
    srad_scalar_input_buffer& in_q0sqr,
    srad_scalar_input_buffer& in_lambda,
    output_buffer<float>& out_j_next);

}
