This directory contains the PL side of the Ours2 SRAD worker-Q0 validation
mapping.

`TopPL.cpp` is instantiated four times. Each CU controls four AIE lanes,
streams four 19x24 halo-padded J tiles per graph firing, stores the returned
16x16 J_next tiles into DDR, and sends one merged partial sum/sum2 statistic
packet to Q0Ctrl.

`Q0Ctrl.cpp` receives the four worker partial statistics, computes the global
q0sqr, and broadcasts the same q0sqr to all TopPL workers before the next SRAD
iteration. Each TopPL embeds that q0sqr into the padding slot of all four lane
tiles it owns.
