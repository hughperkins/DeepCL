#include "StatefulTimer.h"

#include "BackpropWeightsCpu.h"
#include "BackpropWeightsNaive.h"
#include "BackpropWeightsScratchBias.h"

#include "BackpropWeights.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC BackpropWeights *BackpropWeights::instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) {
    if( square( dim.filterSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropWeightsNaive( cl, dim, fn );
    } else {
        return new BackpropWeightsScratchBias( cl, dim, fn );
    }
}
STATIC BackpropWeights *BackpropWeights::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    return new BackpropWeightsCpu( cl, layerDimensions, fn );
}
STATIC BackpropWeights *BackpropWeights::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    if( idx == 0 ) {
        return new BackpropWeightsCpu( cl, layerDimensions, fn );
    }
    if( idx == 1 ) {
        return new BackpropWeightsNaive( cl, layerDimensions, fn );
    }
    if( idx == 2 ) {
        return new BackpropWeightsScratchBias( cl, layerDimensions, fn );
    }
}
BackpropWeights::BackpropWeights( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) :
        dim( layerDimensions ),
        cl( cl ),
        fn( fn ) {
}

VIRTUAL void BackpropWeights::backpropWeights( int batchSize, float learningRate, float *errors, float *results, float *inputData, float *filters, float *biasWeights ) {
    StatefulTimer::timeCheck("BackpropWeights::backprop begin");

//    const float learningMultiplier = learningRate / batchSize / sqrt( dim.outputBoardSize * dim.outputBoardSize );

    CLWrapper *errorsWrapper = cl->wrap( batchSize * dim.outputCubeSize, errors );
    errorsWrapper->copyToDevice();

    CLWrapper *resultsWrapper = cl->wrap( batchSize * dim.outputCubeSize, results );
    resultsWrapper->copyToDevice();

    CLWrapper *inputDataWrapper = cl->wrap( batchSize * dim.inputCubeSize, inputData );
    inputDataWrapper->copyToDevice();

    CLWrapper *weightsWrapper = 0;
    if( debug ) {
        weightsWrapper = cl->wrap(std::max(10000, dim.filtersSize ), filters );
    } else {
        weightsWrapper = cl->wrap( dim.filtersSize, filters );
    }
    weightsWrapper->copyToDevice();

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = cl->wrap( dim.numFilters, biasWeights );
        biasWeightsWrapper->copyToDevice();
    }

    StatefulTimer::timeCheck("BackpropWeights::backprop after copied to device");
    backpropWeights( batchSize, learningRate, errorsWrapper, resultsWrapper, inputDataWrapper, weightsWrapper, biasWeightsWrapper );
    StatefulTimer::timeCheck("BackpropWeights::backprop after call backprop");
    weightsWrapper->copyToHost();
    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
    }
    StatefulTimer::timeCheck("BackpropWeights::backprop after copytohost");

    delete errorsWrapper;
    delete resultsWrapper;
    delete inputDataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }
}

