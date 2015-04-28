// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "StatefulTimer.h"
#include "stringhelper.h"

#include "BackwardCpu.h"
#include "BackwardGpuNaive.h"
#include "BackwardGpuCached.h"

#include "Backward.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC Backward *Backward::instance(OpenCLHelper *cl, LayerDimensions dim ) {
    return new BackwardCpu( cl, dim ); // TODO: remove this line, so uses gpu again
    if( ( dim.inputImageSize - dim.filterSize > 6 ) && square( dim.inputImageSize ) <= cl->getMaxWorkgroupSize() ) {
//        return new BackwardGpuNaive( cl, dim );
        return new BackwardGpuCached( cl, dim );
    } else {
        return new BackwardGpuNaive( cl, dim );
    }
}
STATIC Backward *Backward::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    return new BackwardGpuNaive( cl, layerDimensions );
}
STATIC Backward *Backward::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( idx == 0 ) {
        return new BackwardCpu( cl, layerDimensions );
    }
    if( idx == 1 ) {
        return new BackwardGpuNaive( cl, layerDimensions );
    }
    if( idx == 2 ) {
        return new BackwardGpuCached( cl, layerDimensions );
    }
    throw std::runtime_error("backproperrorsv2::isntancespecifc, index not known: " + toString( idx ) );
}
Backward::Backward( OpenCLHelper *cl, LayerDimensions layerDimensions ) :
        cl( cl ),
        dim( layerDimensions ) {
}
VIRTUAL float * Backward::backward( int batchSize, float *inputData, float *gradOutput, float *filters ) {
    StatefulTimer::timeCheck("Backward::backprop begin");

    CLWrapper *inputDataWrapper = cl->wrap( batchSize * dim.inputCubeSize, inputData );
    inputDataWrapper->copyToDevice();

    CLWrapper *gradOutputWrapper = cl->wrap( batchSize * dim.outputCubeSize, gradOutput );
    gradOutputWrapper->copyToDevice();

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
    int allocatedOutputSize = std::max(5000, outputDataSize );
    float *gradInput = new float[allocatedOutputSize];
    CLWrapper *gradInputWrapper = cl->wrap( allocatedOutputSize, gradInput );

    StatefulTimer::timeCheck("Backward::backprop after copied to device");
    backward( batchSize, inputDataWrapper, gradOutputWrapper, weightsWrapper, gradInputWrapper );
    StatefulTimer::timeCheck("Backward::backprop after call backprop");
    gradInputWrapper->copyToHost();
    StatefulTimer::timeCheck("Backward::backprop after copytohost");

    delete gradInputWrapper;
    delete gradOutputWrapper;
    delete weightsWrapper;
    delete inputDataWrapper;
//    if( dim.biased ) {
//        delete biasWeightsWrapper;
//    }

    return gradInput;
}

