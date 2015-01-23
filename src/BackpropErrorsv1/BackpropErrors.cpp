#include "StatefulTimer.h"

#include "BackpropErrorsCpu.h"
#include "BackpropErrors1.h"
#include "BackpropErrors2.h"

#include "BackpropErrors.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC BackpropErrors *BackpropErrors::instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) {
    if( square( dim.inputBoardSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropErrorsCpu( cl, dim, fn );
    } else {
        return new BackpropErrorsCpu( cl, dim, fn );
    }
}
STATIC BackpropErrors *BackpropErrors::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    return new BackpropErrorsCpu( cl, layerDimensions, fn );
}
STATIC BackpropErrors *BackpropErrors::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    if( idx == 0 ) {
        return new BackpropErrorsCpu( cl, layerDimensions, fn );
    }
    if( idx == 1 ) {
        return new BackpropErrors1( cl, layerDimensions, fn );
    }
    if( idx == 2 ) {
        return new BackpropErrors2( cl, layerDimensions, fn );
    }
}
BackpropErrors::BackpropErrors( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) :
        dim( layerDimensions ),
        cl( cl ),
        fn( fn ) {
}
VIRTUAL float * BackpropErrors::backpropErrors( int batchSize, float *results, float *filters, float *biases, float *errors ) {
    StatefulTimer::timeCheck("BackpropErrors::backprop begin");

    CLWrapper *resultsWrapper = cl->wrap( batchSize * dim.outputCubeSize, results );
    resultsWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        int biasWeightsWrapperSize = dim.numFilters;
        biasWeightsWrapper = cl->wrap( biasWeightsWrapperSize, biases );
        biasWeightsWrapper->copyToDevice();
    }

    CLWrapper *errorsWrapper = cl->wrap( batchSize * dim.outputCubeSize, errors );
    errorsWrapper->copyToDevice();

    int outputDataSize = batchSize * dim.inputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
    int allocatedResultsSize = std::max(5000, outputDataSize );
    float *errorsForUpstream = new float[allocatedResultsSize];
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( allocatedResultsSize, errorsForUpstream );

    StatefulTimer::timeCheck("BackpropErrors::backprop after copied to device");
    backpropErrors( batchSize, resultsWrapper, weightsWrapper, biasWeightsWrapper, errorsWrapper, errorsForUpstreamWrapper );
    StatefulTimer::timeCheck("BackpropErrors::backprop after call backprop");
    errorsForUpstreamWrapper->copyToHost();
    StatefulTimer::timeCheck("BackpropErrors::backprop after copytohost");

    delete errorsForUpstreamWrapper;
    delete errorsWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }

    return errorsForUpstream;
}

