#pragma once

#include <adf.h>
#include "Config.h"

using namespace adf;

using srad_image_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

using srad_mid_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

extern "C" {

void srad_local_q(
    srad_image_input_buffer& in_j,
    output_buffer<float>& out_c);

void srad_coeff_update(
    srad_mid_input_buffer& in_c,
    srad_image_input_buffer& in_j,
    output_buffer<float>& out_j_next);

}
