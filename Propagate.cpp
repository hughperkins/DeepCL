#include <iostream>

#include "Propagate.h"
#include "stringhelper.h"
#include "Propagate1.h"
#include "Propagate2.h"
#include "Propagate3.h"
#include "StatefulTimer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC Propagate *Propagate::instance(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
        return new Propagate1( cl, layerDimensions, fn );
}
Propagate::Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const*fn ) :
        dim( layerDimensions ),
        cl( cl ),
        fn( fn ) {
}
VIRTUAL float * Propagate::propagate( int batchSize, float *inputData, float *filters, float *biases ) {
    StatefulTimer::timeCheck("Propagate::propagate begin");
    int inputDataSize = batchSize * dim.inputPlanes * square( dim.inputBoardSize );
    CLWrapper *dataWrapper = cl->wrap( inputDataSize, inputData );
    dataWrapper->copyToDevice();

    int weightsSize = dim.inputPlanes * dim.numFilters * square( dim.filterSize );
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        int biasWeightsWrapperSize = dim.numFilters;
        biasWeightsWrapper = cl->wrap( biasWeightsWrapperSize, biases );
        biasWeightsWrapper->copyToDevice();
    }

    int outputDataSize = batchSize * dim.numFilters * square( dim.outputBoardSize );
//    cout << " batchsize " << batchSize << " numfilters " << dim.numFilters << " outputboardsize "
//        << dim.outputBoardSize << " outputdatasize " << outputDataSize << endl;
    int allocatedResultsSize = outputDataSize; // std::max(5000, outputDataSize );
    float *results = new float[allocatedResultsSize];
    CLWrapper *resultsWrapper = cl->wrap( allocatedResultsSize, results );

    StatefulTimer::timeCheck("Propagate::propagate after copied to device");
    propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper,
            resultsWrapper );
    StatefulTimer::timeCheck("Propagate::propagate after call propagate");
    resultsWrapper->copyToHost();
    StatefulTimer::timeCheck("Propagate::propagate after copytohost");

//    for( int i = 0; i < 20; i++ ) {
//        cout << "results[" << i << "]=" << results[i] << endl;
//    }

    delete dataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }
    delete resultsWrapper;

    return results;
}

