This directory contains the PL side of the Ours SRAD mapping.

Current board path:

- `TopPL.cpp` is instantiated six times.
- The six CUs form three row blocks by two column workers. Each `TopPL` owns
  16 AIE lanes and one contiguous 2000-column half image over a 1334-row center
  block.
- `TopPL_0/1` handle center rows 0..1333, `TopPL_2/3` handle rows
  1334..2667, and `TopPL_4/5` handle padded rows 2668..4001. Even worker IDs
  handle columns 0..1999; odd worker IDs handle columns 2000..3999.
- Each worker streams one north context row, 1334 center rows, and two south
  context rows to AIE. Internal boundaries such as rows 1333/1334 and
  2667/2668 are handled by rereading those DDR rows as halo context.
- Padding rows beyond the real 4000-row image are consumed to keep the AIE stream
  shape uniform but are not written back and are not included in `sum/sum2`.
- `Q0Ctrl.cpp` aggregates the six local `sum/sum2` reports, computes one global
  `q0sqr` over `kBoardPixels`, and broadcasts that value back to all TopPL CUs.
