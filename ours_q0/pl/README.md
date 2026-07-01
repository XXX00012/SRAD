This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated once.
- Each SRAD iteration has two image reads. `TopPL` first streams the current
  image through the AIE row-stats kernel, accumulates returned row `sum/sum2`,
  and computes `q0sqr`.
- `TopPL` then streams the same current image again, with the computed `q0sqr`
  embedded in each row, through the existing two-kernel update pipeline and
  stores the returned `J_next` image to DDR.
- `Q0Ctrl.cpp` is kept for future multi-`TopPL` aggregation paths, but it is
  not used by the current single-`TopPL` build.
