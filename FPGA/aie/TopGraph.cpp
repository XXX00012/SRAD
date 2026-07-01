#include "TopGraph.h"

#include <cstddef>

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>
#endif

GraphFpgaV5 graphFpgaV5("fpga_v5");

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
    std::printf("[aiesim] GraphFpgaV5 srad_fpga stream/tile baseline\n");

    graphFpgaV5.init();
    graphFpgaV5.run(1);
    graphFpgaV5.wait();
    graphFpgaV5.end();
    return 0;
}
#endif
