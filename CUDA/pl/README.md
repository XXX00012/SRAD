This directory contains the PL bridge code for the current CUDA-style SRAD AIE
mapping.

`TopPL.cpp` intentionally defines four HLS kernels from one source file:

- `PreparePL`: DDR image block -> `gpu_prepare`, and writes sums/sums2 to DDR.
- `ReducePL`: DDR sums/sums2 -> `gpu_reduce`, and writes per-block partials.
- `CoeffPL`: DDR image + q0 -> `gpu_srad`, and writes dN/dS/dW/dE/c planes.
- `UpdatePL`: DDR image + planes -> `gpu_srad2`, and writes next image.

`Q0Ctrl.cpp` is legacy row-stream code and is not built by the active Makefile.
