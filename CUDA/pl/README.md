# PLIO movers

This OpenCL-v0-faithful baseline uses PL HLS movers for DDR <-> PLIO
transport. Global arrays use the OpenCL linear layout `row + Nr * col`.

The AIE pipeline follows the OpenCL v0 stage boundaries: `prepare_kernel`
materializes `d_sums` and `d_sums2`, `reduce_kernel` preserves the v0
256-thread reduction tree, the host computes `q0sqr`, and the coefficient and
update stages run as the v0 `srad_kernel` and `srad2_kernel` equivalents. PL
movers only stream, tile, halo-pack, and store the global arrays.
