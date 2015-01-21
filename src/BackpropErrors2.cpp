#include "BackpropErrors2.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropErrors2::BackpropErrors2( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) :
        BackpropErrors( cl, dim, fn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();
    kernel = cl->buildKernel( "backproperrors.cl", "calcErrorsForUpstreamCached", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropErrors2::~BackpropErrors2() {
    delete kernel;
}
VIRTUAL void BackpropErrors2::backpropErrors( int batchSize, 
        CLWrapper *resultsWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {

    StatefulTimer::instance()->timeCheck("BackpropErrors2 start" );
    kernel
        ->in( batchSize )
        ->in( errorsWrapper )
        ->in( weightsWrapper )
        ->out( errorsForUpstreamWrapper )
        ->localFloats( square( dim.outputBoardSize ) )
        ->localFloats( square( dim.filterSize ) );

    int numWorkgroups = batchSize * dim.inputPlanes;
    int workgroupSize = square( dim.inputBoardSize );
    workgroupSize = std::max( 32, workgroupSize ); // no point in wasting cores...
    int globalSize = numWorkgroups * workgroupSize;

//    int globalSize = batchSize * dim.inputCubeSize;
//    int workgroupsize = cl->getMaxWorkgroupSize();
//    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    std::cout << "BackpropErrors2 workgroupsize " << workgroupSize << " globalsize " << globalSize << std::endl;
    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropErrors2 end" );
}

