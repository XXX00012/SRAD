#pragma once

#include <adf.h>
#include "Config.h"

using namespace adf;

using srad_local_q_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kLocalQMarginElems>>;

using srad_update_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>,
    adf::margin<srad_cfg::kUpdateMarginElems>>;

using srad_mid_input_buffer = input_buffer<
    float,
    adf::extents<adf::inherited_extent>>;

extern "C" {

void srad_local_q(srad_local_q_input_buffer& in_j,
                  output_buffer<float>& out_c);

void srad_coeff_update(srad_mid_input_buffer& in_c,
                       srad_update_input_buffer& in_j,
                       output_buffer<float>& out_j_next);

}
