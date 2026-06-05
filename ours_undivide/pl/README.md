This directory contains the PL q0sqr reduction engine for the Ours SRAD
mapping.

`TopPL.cpp` scans the configured float32 image from DDR, computes q0sqr,
streams the q0sqr/J/J_update/lambda packets into AIE through PLIO, and stores
the returned J_next stream to DDR.
