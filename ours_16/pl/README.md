This directory contains the PL side of the Ours16 SRAD worker-Q0 validation
mapping.

`TopPL.cpp` is currently instantiated twice. Each CU controls eight AIE lanes,
streams eight 19x24 halo-padded J tiles per graph firing, stores the returned
16x16 J_next tiles into DDR, and sends one merged partial sum/sum2 statistic
packet to Q0Ctrl.

`Q0Ctrl.cpp` receives one worker partial statistic per TopPL, computes the global
q0sqr, and broadcasts the same q0sqr to all TopPL workers before the next SRAD
iteration. Each TopPL embeds that q0sqr into the padding slot of all lane tiles
it owns. The generated templates are intended to scale this same pattern by
adding more eight-lane TopPL workers.
