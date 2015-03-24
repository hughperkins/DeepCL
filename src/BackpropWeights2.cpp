// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "StatefulTimer.h"

#include "stringhelper.h"

#include "BackpropWeights2.h"
#include "BackpropWeights2Cpu.h"
#include "BackpropWeights2Naive.h"
#include "BackpropWeights2Scratch.h"
#include "BackpropWeights2ScratchLarge.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeights2::BackpropWeights2( OpenCLHelper *cl, LayerDimensions layerDimensions ) :
        cl( cl ),
        dim( layerDimensions ) {
}
STATIC BackpropWeights2 *BackpropWeights2::instance(OpenCLHelper *cl, LayerDimensions dim ) {
    if( dim.inputImageSize - dim.filterSize < 4 ) {
        return new BackpropWeights2Naive( cl, dim );
    }
    if( square( dim.filterSize ) <= cl->getMaxWorkgroupSize() 
            && dim.inputImageSize <= 32 ) { // if inputimagesize too big, we run out of local memory
        return new BackpropWeights2Scratch( cl, dim );
    } else if( square( dim.filterSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropWeights2ScratchLarge( cl, dim );
    } else {
        return new BackpropWeights2Naive( cl, dim );
    }

//    return new BackpropWeights2ScratchLarge( cl, dim );

//    if( square( dim.filterSize ) <= cl->getMaxWorkgroupSize() 
//            && dim.inputImageSize <= 32 ) { // if inputimagesize too big, we run out of local memory
//        return new BackpropWeights2Scratch( cl, dim );
//    } else {
//        return new BackpropWeights2Naive( cl, dim );
//    }
}
STATIC BackpropWeights2 *BackpropWeights2::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    return new BackpropWeights2ScratchLarge( cl, layerDimensions );
}
STATIC BackpropWeights2 *BackpropWeights2::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( idx == 0 ) {
        return new BackpropWeights2Cpu( cl, layerDimensions );
    }
    if( idx == 1 ) {
        return new BackpropWeights2Naive( cl, layerDimensions );
    }
    if( idx == 2 ) {
        return new BackpropWeights2Scratch( cl, layerDimensions );
    }
    if( idx == 3 ) {
        return new BackpropWeights2ScratchLarge( cl, layerDimensions );
    }
    throw std::runtime_error("BackpropWeights::instanceSpecific doesnt handle idx " + toString(idx) );
}

VIRTUAL void BackpropWeights2::backpropWeights( int batchSize, float learningRate, float *derivLossBySum, float *inputData, float *filters, float *biasWeights ) {
    StatefulTimer::timeCheck("BackpropWeights2::backprop begin");

//    const float learningMultiplier = learningRate / batchSize / sqrt( dim.outputImageSize * dim.outputImageSize );

    int resultsSize = batchSize * dim.outputCubeSize;
    CLWrapper *derivLossBySumWrapper = cl->wrap( resultsSize, derivLossBySum );
    derivLossBySumWrapper->copyToDevice();

    int inputSize = batchSize * dim.inputCubeSize;
    CLWrapper *inputDataWrapper = cl->wrap( inputSize, inputData );
    inputDataWrapper->copyToDevice();

    CLWrapper *weightsWrapper = 0;
    int weightsSize = debug ? std::max(10000, dim.filtersSize ) : dim.filtersSize;
    weightsWrapper = cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

//    cout << "backpropweights2::backpropweights resultsSize=" << resultsSize << " inputSize=" << inputSize << 
//        " weightSize=" << weightsSize << endl;

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = cl->wrap( dim.numFilters, biasWeights );
        biasWeightsWrapper->copyToDevice();
    }

    StatefulTimer::timeCheck("BackpropWeights2::backprop after copied to device");
    backpropWeights( batchSize, learningRate, derivLossBySumWrapper, inputDataWrapper, weightsWrapper, biasWeightsWrapper );
    StatefulTimer::timeCheck("BackpropWeights2::backprop after call backprop");
    weightsWrapper->copyToHost();
    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
    }
    StatefulTimer::timeCheck("BackpropWeights2::backprop after copytohost");

    delete derivLossBySumWrapper;
    delete inputDataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }
}

float BackpropWeights2::learningRateToMultiplier( int batchSize, float rate ) {
//        float multiplier = rate / batchSize / sqrt( dim.outputImageSize );
    float multiplier = rate;
//    std::cout << "rate " << rate << " multiplier " << multiplier << std::endl;
    return multiplier;
}


