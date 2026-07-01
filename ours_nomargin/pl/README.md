This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated once.
- `TopPL` scans the input image locally, computes `q0sqr` from `sum/sum2`,
  streams the AIE row data, and stores the returned `J_next` image to DDR.
- `Q0Ctrl.cpp` is kept for future multi-`TopPL` aggregation paths, but it is
  not used by the current single-`TopPL` build.
