#pragma once

#include <adf.h>
#include "Config.h"

using namespace adf;

using srad_float_input_stream = input_stream<float>;
using srad_float_output_stream = output_stream<float>;

extern "C" {

void srad_fpga_v5(srad_float_input_stream* in_params,
                  srad_float_input_stream* in_compute,
                  srad_float_output_stream* out_index,
                  srad_float_output_stream* out_value);

}
