# SRAD Ours_400 PL Kernels

`TopPL.cpp` is the active worker kernel for this build. The top-level
Makefile instantiates it four times:

- `TopPL_0`: AIE lanes 0..49
- `TopPL_1`: AIE lanes 50..99
- `TopPL_2`: AIE lanes 100..149
- `TopPL_3`: AIE lanes 150..199

Each TopPL CU has 50 input PLIO streams and 25 `pktmerge<2>` output streams.
`Q0Ctrl.cpp` reduces the four CU partial sums and broadcasts `q0sqr`.
