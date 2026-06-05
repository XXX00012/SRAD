#include "TopGraph.h"

#include <cstddef>

GraphFpgaV5 graphFpgaV5("fpga_v5");

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>

int main() {
    std::printf("[aiesim] GraphFpgaV5 PLIO float fused reduction + compute\n");
    std::fflush(stdout);
    graphFpgaV5.init();
    graphFpgaV5.run(1);
    graphFpgaV5.wait();
    graphFpgaV5.end();
    std::printf("[aiesim] wrote ./data/plio_fpga_j_next.txt\n");
    return 0;
}
#endif
