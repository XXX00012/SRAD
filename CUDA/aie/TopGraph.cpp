#include <TopGraph.h>

GraphPrepare graphPrepare("gpu_prepare");
GraphReduce graphReduce("gpu_reduce");
GraphSradCoeff graphSradCoeff("gpu_srad");
GraphSradUpdate graphSradUpdate("gpu_srad2");

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
    graphPrepare.init();
    graphReduce.init();
    graphSradCoeff.init();
    graphSradUpdate.init();

    graphPrepare.run(1);
    graphReduce.run(1);
    graphSradCoeff.run(1);
    graphSradUpdate.run(1);

    graphSradUpdate.end();
    graphSradCoeff.end();
    graphReduce.end();
    graphPrepare.end();
    return 0;
}
#endif
