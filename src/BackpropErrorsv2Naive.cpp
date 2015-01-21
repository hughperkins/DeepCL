#include "StatefulTimer.h"

#include "BackpropErrorsv2Naive.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropErrorsv2Naive::BackpropErrorsv2Naive( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn ) :
        BackpropErrorsv2( cl, dim, upstreamFn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();
    options += " -D " + upstreamFn->getDefineName();
    kernel = cl->buildKernel( "backproperrorsv2.cl", "calcErrorsForUpstream", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropErrorsv2Naive::~BackpropErrorsv2Naive() {
    delete kernel;
}
VIRTUAL void BackpropErrorsv2Naive::backpropErrors( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *errorsWrapper, CLWrapper *weightsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive start" );

    kernel
       ->in( batchSize )
       ->in( inputDataWrapper )
        ->in( errorsWrapper )
       ->in( weightsWrapper )
        ->out( errorsForUpstreamWrapper );

    int globalSize = batchSize * dim.inputCubeSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive end" );
}

