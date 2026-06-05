This FPGA-v5-style AIE baseline does not use a PL kernel.

q0sqr is computed inside `aie/srad_fpga.cc` Phase 1, then reused by the fused
Phase 2. The `pl/` directory is present only to keep the project layout aligned
with the existing SRAD baselines.
