#include "TopGraph.h"

#include <cstddef>

GraphFpgaV5PLQ0 graphFpgaV5PLQ0("fpga_v5_plq0");

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>

int main() {
    std::printf("[aiesim] GraphFpgaV5PLQ0 PLIO float phase2 with PL q0sqr fixture\n");
    std::fflush(stdout);
    graphFpgaV5PLQ0.init();
    graphFpgaV5PLQ0.run(1);
    graphFpgaV5PLQ0.wait();
    graphFpgaV5PLQ0.end();
    std::printf("[aiesim] wrote ./data/plio_fpga_pl_j_next.txt\n");
    return 0;
}
#endif
