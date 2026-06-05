This directory contains the PL q0sqr reduction engine for the Ours SRAD
mapping.

`TopPL.cpp` scans the configured float32 image, computes q0sqr, and writes a
float32 scalar packet to the `q0_debug` M_AXI buffer. The host copies this
PL-computed packet into the AIE graph through the normal `in_q0sqr` GMIO.
