// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"

#include "stringhelper.h"

#include "BackpropWeights2.h"
#include "BackpropWeights2Cpu.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC BackpropWeights2 *BackpropWeights2::instance(OpenCLHelper *cl, LayerDimensions dim ) {
//    if( square( dim.filterSize ) <= cl->getMaxWorkgroupSize() ) {
        return new BackpropWeights2Cpu( cl, dim );
//    } else {
//        return new BackpropWeights2ScratchBias( cl, dim, fn );
//    }
}
STATIC BackpropWeights2 *BackpropWeights2::instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    return new BackpropWeights2Cpu( cl, layerDimensions );
}
STATIC BackpropWeights2 *BackpropWeights2::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( idx == 0 ) {
        return new BackpropWeights2Cpu( cl, layerDimensions );
    }
//    if( idx == 1 ) {
//        return new BackpropWeights2Naive( cl, layerDimensions, fn );
//    }
//    if( idx == 2 ) {
//        return new BackpropWeights2ScratchBias( cl, layerDimensions, fn );
//    }
    throw std::runtime_error("BackpropWeights::instanceSpecific doesnt handle idx " + toString(idx) );
}
BackpropWeights2::BackpropWeights2( OpenCLHelper *cl, LayerDimensions layerDimensions ) :
        dim( layerDimensions ),
        cl( cl ) {
}

VIRTUAL void BackpropWeights2::backpropWeights( int batchSize, float learningRate, float *derivLossBySum, float *inputData, float *filters, float *biasWeights ) {
    StatefulTimer::timeCheck("BackpropWeights2::backprop begin");

//    const float learningMultiplier = learningRate / batchSize / sqrt( dim.outputBoardSize * dim.outputBoardSize );

    CLWrapper *derivLossBySumWrapper = cl->wrap( batchSize * dim.outputCubeSize, derivLossBySum );
    derivLossBySumWrapper->copyToDevice();

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
//        float multiplier = rate / batchSize / sqrt( dim.outputBoardSize );
    float multiplier = rate;
//    std::cout << "rate " << rate << " multiplier " << multiplier << std::endl;
    return multiplier;
}


