// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "StatefulTimer.h"
#include "stringhelper.h"

#include "BackpropErrorsv2Cpu.h"
#include "BackpropErrorsv2Naive.h"
#include "BackpropErrorsv2Cached.h"

#include "BackpropErrorsv2.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC BackpropErrorsv2 *BackpropErrorsv2::instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn ) {
    if( square( dim.inputBoardSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropErrorsv2Naive( cl, dim, upstreamFn );
    } else {
        return new BackpropErrorsv2Naive( cl, dim, upstreamFn );
    }
}
STATIC BackpropErrorsv2 *BackpropErrorsv2::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn ) {
    return new BackpropErrorsv2Naive( cl, layerDimensions, upstreamFn );
}
STATIC BackpropErrorsv2 *BackpropErrorsv2::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn ) {
    if( idx == 0 ) {
        return new BackpropErrorsv2Cpu( cl, layerDimensions, upstreamFn );
    }
    if( idx == 1 ) {
        return new BackpropErrorsv2Naive( cl, layerDimensions, upstreamFn );
    }
    if( idx == 2 ) {
        return new BackpropErrorsv2Cached( cl, layerDimensions, upstreamFn );
    }
    throw std::runtime_error("backproperrorsv2::isntancespecifc, index not known: " + toString( idx ) );
}
BackpropErrorsv2::BackpropErrorsv2( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn ) :
        dim( layerDimensions ),
        cl( cl ),
        upstreamFn( upstreamFn ) {
}
VIRTUAL float * BackpropErrorsv2::backpropErrors( int batchSize, float *inputData, float *errors, float *filters ) {
    StatefulTimer::timeCheck("BackpropErrorsv2::backprop begin");

    CLWrapper *inputDataWrapper = cl->wrap( batchSize * dim.inputCubeSize, inputData );
    inputDataWrapper->copyToDevice();

    CLWrapper *errorsWrapper = cl->wrap( batchSize * dim.outputCubeSize, errors );
    errorsWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

//    CLWrapper *biasWeightsWrapper = 0;
//    if( dim.biased ) {
//        int biasWeightsWrapperSize = dim.numFilters;
//        biasWeightsWrapper = cl->wrap( biasWeightsWrapperSize, biases );
//        biasWeightsWrapper->copyToDevice();
//    }

    int outputDataSize = batchSize * dim.inputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
    int allocatedResultsSize = std::max(5000, outputDataSize );
    float *errorsForUpstream = new float[allocatedResultsSize];
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( allocatedResultsSize, errorsForUpstream );

    StatefulTimer::timeCheck("BackpropErrorsv2::backprop after copied to device");
    backpropErrors( batchSize, inputDataWrapper, errorsWrapper, weightsWrapper, errorsForUpstreamWrapper );
    StatefulTimer::timeCheck("BackpropErrorsv2::backprop after call backprop");
    errorsForUpstreamWrapper->copyToHost();
    StatefulTimer::timeCheck("BackpropErrorsv2::backprop after copytohost");

    delete errorsForUpstreamWrapper;
    delete errorsWrapper;
    delete weightsWrapper;
    delete inputDataWrapper;
//    if( dim.biased ) {
//        delete biasWeightsWrapper;
//    }

    return errorsForUpstream;
}

