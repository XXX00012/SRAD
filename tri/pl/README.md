This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated once.
- `TopPL` scans the input image locally, computes `q0sqr` from `sum/sum2`,
  streams the AIE row data, and stores the returned `J_next` image to DDR.
- AIE K3 appends per-row `sum/sum2` to the returned row padding; `TopPL`
  accumulates those row stats for the next iteration's `q0sqr`.
- `Q0Ctrl.cpp` is kept for future multi-`TopPL` aggregation paths, but it is
  not used by the current single-`TopPL` build.
