This directory contains the PL side of the Ours SRAD worker-Q0 validation
mapping.

`TopPLWorker.cpp` is instantiated four times. Each worker handles one AIE
lane, streams one 19x24 halo-padded J tile per graph firing, stores the
returned 16x16 J_next tile into DDR, and sends partial sum/sum2 statistics.

`Q0Ctrl.cpp` receives the four partial statistics, computes the global q0sqr,
and broadcasts the same q0sqr to all workers before the next SRAD iteration.
