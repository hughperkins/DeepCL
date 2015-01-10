#include "BackpropWeightsNaive.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsNaive::BackpropWeightsNaive( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) :
        BackpropWeights( cl, dim, fn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();
    options += " -D " + fn->getDefineName();
    kernel = cl->buildKernel( "backpropweights.cl", "backprop_floats", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropWeightsNaive::~BackpropWeightsNaive() {
    delete kernel;
}
VIRTUAL void BackpropWeightsNaive::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsNaive start" );

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )->in( dim.inputPlanes )->in( dim.numFilters )
       ->in( dim.inputBoardSize )->in( dim.filterSize )->in( dim.outputBoardSize )->in( dim.padZeros ? 1 : 0 )       
        ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->inout( weightsWrapper );
    if( dim.biased ) {
        kernel->inout( biasWeightsWrapper );
    }

    int globalSize = dim.filtersSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeightsNaive end" );
}

