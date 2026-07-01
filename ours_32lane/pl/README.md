This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated twice.
- Each `TopPL` CU owns 16 AIE lanes and one contiguous 2000-column half of the
  4000-column board image. `TopPL_0` handles columns 0..1999 and AIE lanes
  0..15; `TopPL_1` handles columns 2000..3999 and AIE lanes 16..31.
- Each `TopPL` keeps DDR accesses row-major and contiguous over its 2000-column
  worker range before distributing data to its 16 PLIO streams.
- `Q0Ctrl.cpp` aggregates the two local `sum/sum2` reports, computes one global
  `q0sqr` over `kBoardPixels`, and broadcasts that value back to both TopPL CUs.
