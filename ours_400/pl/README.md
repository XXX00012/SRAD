This directory contains the PL q0sqr reduction engine for the Ours SRAD
mapping.

`TopPL.cpp` scans the configured float32 image from DDR, computes q0sqr,
embeds q0sqr into the first unused padding column of each 19x24 J tile,
streams that J tile into the AIE graph through one PLIO, and stores the
AIE J_next PLIO stream back to DDR.
