#include "TopGraph.h"

#if defined(__AIESIM__) || defined(__X86SIM__)
#include <cstdio>
#endif

GraphOursPLQ0 graphOursPLQ0("ours_plq0");

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
    std::printf("[aiesim] ours 192-lane row-stream: %d rows/lane, row width %d\n",
                srad_cfg::kGraphRowsSim,
                srad_cfg::kRowPhysElems);

    graphOursPLQ0.init();
    graphOursPLQ0.run(srad_cfg::kGraphRowsSim);
    graphOursPLQ0.end();
    return 0;
}
#endif
