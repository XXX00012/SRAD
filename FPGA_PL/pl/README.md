This directory contains the PL q0sqr reduction engine for the FPGA_PL SRAD
baseline.

`TopPL.cpp` builds two transport kernels. `LoadFpgaV5PLQ0` scans the
configured float32 image, computes q0sqr in PL, and streams the image, q0sqr
packet, and lambda packet to AIE through PLIO. `StoreFpgaV5PLQ0` streams the
AIE result back to DDR.
