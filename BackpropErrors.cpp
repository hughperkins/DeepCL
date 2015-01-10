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

STATIC BackpropErrors *BackpropErrors::instance(OpenCLHelper *cl, LayerDimensions dim ) {
    if( square( dim.inputBoardSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropErrors1( cl, dim );
    } else {
        return new BackpropErrors1( cl, dim );
    }
}
STATIC BackpropErrors *BackpropErrors::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    return new BackpropErrors2( cl, layerDimensions );
}
STATIC BackpropErrors *BackpropErrors::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( idx == 0 ) {
        return new BackpropErrorsCpu( cl, layerDimensions );
    }
    if( idx == 1 ) {
        return new BackpropErrors1( cl, layerDimensions );
    }
    if( idx == 2 ) {
        return new BackpropErrors2( cl, layerDimensions );
    }
}
BackpropErrors::BackpropErrors( OpenCLHelper *cl, LayerDimensions layerDimensions ) :
        dim( layerDimensions ),
        cl( cl ) {
}
VIRTUAL float * BackpropErrors::backpropErrors( int batchSize, float *filters, float *biases, float *errors ) {
    StatefulTimer::timeCheck("BackpropErrors::backprop begin");

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
    backpropErrors( batchSize, weightsWrapper, biasWeightsWrapper, errorsWrapper, errorsForUpstreamWrapper );
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

