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

using srad_image_output_buffer = output_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

extern "C" {

void srad_fpga_v5_phase2(
    srad_image_input_buffer& in_compute,
    srad_scalar_input_buffer& in_q0sqr,
    srad_scalar_input_buffer& in_lambda,
    srad_image_output_buffer& out_j_next);

}
