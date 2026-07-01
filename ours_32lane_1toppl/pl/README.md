This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated once.
- `TopPL` scans the input image locally, computes `q0sqr` from `sum/sum2`,
  streams the AIE row data, and stores the returned `J_next` image to DDR.
- One `TopPL` controls 32 AIE row-stream lanes. The 4000-column board image is
  split into 32 strips of 125 columns, so one strip batch covers a full row.
- `Q0Ctrl.cpp` is kept for future multi-`TopPL` aggregation paths, but it is
  not used by the current single-`TopPL` build.
