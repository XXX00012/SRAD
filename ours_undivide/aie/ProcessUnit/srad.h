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

void srad_undivide_fused(
    srad_image_input_buffer& in_j,
    srad_image_input_buffer& in_j_update,
    srad_scalar_input_buffer& in_q0sqr,
    srad_scalar_input_buffer& in_lambda,
    output_buffer<float>& out_j_next);

}
