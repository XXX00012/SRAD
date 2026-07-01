This directory contains the PL transport shims for the existing
`srad_fpga_v5` AIE kernel.

`LoadFpgaV5` streams the input image in the exact order expected by the AIE
kernel: first the full image for q0sqr reduction, then each halo tile for the
update pass. `StoreFpgaV5` receives `(index,value)` pairs from AIE and writes
valid updates back to DDR.
