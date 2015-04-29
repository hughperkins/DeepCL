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
        dim( layerDimensions ),
        debug( false ) {
}
STATIC BackpropWeights2 *BackpropWeights2::instance(OpenCLHelper *cl, LayerDimensions dim ) {
    return new BackpropWeights2Naive( cl, dim );
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

VIRTUAL void BackpropWeights2::calcGradWeights( int batchSize, float learningRate, float *gradOutput, float *inputs, float *gradWeights, float *gradBiasWeights ) {
    StatefulTimer::timeCheck("BackpropWeights2::backprop begin");

//    const float learningMultiplier = learningRate / batchSize / sqrt( dim.outputImageSize * dim.outputImageSize );

    int outputSize = batchSize * dim.outputCubeSize;
    CLWrapper *gradOutputWrapper = cl->wrap( outputSize, gradOutput );
    gradOutputWrapper->copyToDevice();

    int inputSize = batchSize * dim.inputCubeSize;
    CLWrapper *inputDataWrapper = cl->wrap( inputSize, inputs );
    inputDataWrapper->copyToDevice();

    CLWrapper *gradWeightsWrapper = 0;
    int gradWeightsSize = debug ? std::max(10000, dim.filtersSize ) : dim.filtersSize;
    gradWeightsWrapper = cl->wrap( gradWeightsSize, gradWeights );
    gradWeightsWrapper->copyToDevice();

//    cout << "backpropgradWeights2::backpropgradWeights outputSize=" << outputSize << " inputSize=" << inputSize << 
//        " weightSize=" << gradWeightsSize << endl;

    CLWrapper *gradBiasWeightsWrapper = 0;
    if( dim.biased ) {
        gradBiasWeightsWrapper = cl->wrap( dim.numFilters, gradBiasWeights );
        gradBiasWeightsWrapper->copyToDevice();
    }

    StatefulTimer::timeCheck("BackpropWeights2::backprop after copied to device");
    calcGradWeights( batchSize, learningRate, gradOutputWrapper, inputDataWrapper, gradWeightsWrapper, gradBiasWeightsWrapper );
    StatefulTimer::timeCheck("BackpropWeights2::backprop after call backprop");
    gradWeightsWrapper->copyToHost();
    if( dim.biased ) {
        gradBiasWeightsWrapper->copyToHost();
    }
    StatefulTimer::timeCheck("BackpropWeights2::backprop after copytohost");

    delete gradOutputWrapper;
    delete inputDataWrapper;
    delete gradWeightsWrapper;
    if( dim.biased ) {
        delete gradBiasWeightsWrapper;
    }
}

float BackpropWeights2::learningRateToMultiplier( int batchSize, float rate ) {
//        float multiplier = rate / batchSize / sqrt( dim.outputImageSize );
    float multiplier = rate;
//    std::cout << "rate " << rate << " multiplier " << multiplier << std::endl;
    return multiplier;
}


