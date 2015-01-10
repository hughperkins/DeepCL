#include "BackpropWeightsScratchBias.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsScratchBias::BackpropWeightsScratchBias( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) :
        BackpropWeights( cl, dim, fn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();
    options += " -D " + fn->getDefineName();
    kernel = cl->buildKernel( "backpropweights.cl", "backprop_floats_withscratch_dobias", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropWeightsScratchBias::~BackpropWeightsScratchBias() {
    delete kernel;
}
VIRTUAL void BackpropWeightsScratchBias::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsScratchAndBias start" );
    int workgroupsize = std::max( 32, square( dim.filterSize ) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

    const float learningMultiplier = learningRate / batchSize / sqrt( dim.outputBoardSize * dim.outputBoardSize );

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
       
        ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->inout( weightsWrapper );

        if( dim.biased ) {
            kernel->inout( biasWeightsWrapper );
        }

        kernel->localFloats( square( dim.inputBoardSize ) )
        ->localFloats( square( dim.outputBoardSize ) )
        ->localFloats( square( dim.outputBoardSize ) );
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeightsScratchAndBias end" );
}

