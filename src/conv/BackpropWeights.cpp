// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "util/StatefulTimer.h"

#include "util/stringhelper.h"

#include "BackpropWeights.h"
#include "BackpropWeightsCpu.h"
#include "BackpropWeightsNaive.h"
#include "BackpropWeightsScratch.h"
#include "BackpropWeightsScratchLarge.h"
#include "BackpropWeightsIm2Col.h"
#include "BackpropWeightsAuto.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeights::BackpropWeights(EasyCL *cl, LayerDimensions layerDimensions) :
        cl(cl),
        dim(layerDimensions),
        debug(false) {
}
STATIC BackpropWeights *BackpropWeights::instance(EasyCL *cl, LayerDimensions dim) {
    return new BackpropWeightsAuto(cl, dim);
//    if(dim.inputSize - dim.filterSize < 4) {
//        return new BackpropWeightsNaive(cl, dim);
//    }
//    if(square(dim.filterSize) <= cl->getMaxWorkgroupSize() 
//            && dim.inputSize <= 32) { // if inputimagesize too big, we run out of local memory
//        return new BackpropWeightsScratch(cl, dim);
//    } else if(square(dim.filterSize) <= cl->getMaxWorkgroupSize()) {
//        return new BackpropWeightsScratchLarge(cl, dim);
//    } else {
//        return new BackpropWeightsNaive(cl, dim);
//    }
}
STATIC int BackpropWeights::getNumImplementations() {
    return 5;
}
STATIC bool BackpropWeights::plausiblyOptimal(int index, int batchSize, LayerDimensions dim) {
    if(index == 0) { 
        return false;
    }
    if(index >= 5) {
        return false;
    }
    return true;
}
STATIC BackpropWeights *BackpropWeights::instanceForTest(EasyCL *cl, LayerDimensions layerDimensions) {
    return new BackpropWeightsScratchLarge(cl, layerDimensions);
}
STATIC BackpropWeights *BackpropWeights::instanceSpecific(int idx, EasyCL *cl, LayerDimensions layerDimensions) {
    if(idx == -1) {
        return new BackpropWeightsAuto(cl, layerDimensions);
    }
    if(idx == 0) {
        return new BackpropWeightsCpu(cl, layerDimensions);
    }
    if(idx == 1) {
        return new BackpropWeightsNaive(cl, layerDimensions);
    }
    if(idx == 2) {
        return new BackpropWeightsScratch(cl, layerDimensions);
    }
    if(idx == 3) {
        return new BackpropWeightsScratchLarge(cl, layerDimensions);
    }
    if(idx == 4) {
        return new BackpropWeightsIm2Col(cl, layerDimensions);
    }
    throw std::runtime_error("BackpropWeights::instanceSpecific doesnt handle idx " + toString(idx));
}

VIRTUAL void BackpropWeights::calcGradWeights(int batchSize, float *gradOutput, float *inputs, float *gradWeights, float *gradBias) {
    StatefulTimer::timeCheck("BackpropWeights::backprop begin");

//    const float learningMultiplier = learningRate / batchSize / sqrt(dim.outputSize * dim.outputSize);

    int outputNumElements = batchSize * dim.outputCubeSize;
    CLWrapper *gradOutputWrapper = cl->wrap(outputNumElements, gradOutput);
    gradOutputWrapper->copyToDevice();

    int inputNumElements = batchSize * dim.inputCubeSize;
    CLWrapper *inputDataWrapper = cl->wrap(inputNumElements, inputs);
    inputDataWrapper->copyToDevice();

    CLWrapper *gradWeightsWrapper = 0;
    int gradWeightsSize = debug ? std::max(10000, dim.filtersSize) : dim.filtersSize;
    gradWeightsWrapper = cl->wrap(gradWeightsSize, gradWeights);
    gradWeightsWrapper->copyToDevice();

    CLWrapper *gradBiasWrapper = 0;
    if(dim.biased) {
        gradBiasWrapper = cl->wrap(dim.numFilters, gradBias);
        gradBiasWrapper->copyToDevice();
    }

    StatefulTimer::timeCheck("BackpropWeights::backprop after copied to device");
    calcGradWeights(batchSize, gradOutputWrapper, inputDataWrapper, gradWeightsWrapper, gradBiasWrapper);
    StatefulTimer::timeCheck("BackpropWeights::backprop after call backprop");
    gradWeightsWrapper->copyToHost();
    if(dim.biased) {
        gradBiasWrapper->copyToHost();
    }
    StatefulTimer::timeCheck("BackpropWeights::backprop after copytohost");

    delete gradOutputWrapper;
    delete inputDataWrapper;
    delete gradWeightsWrapper;
    if(dim.biased) {
        delete gradBiasWrapper;
    }
}

float BackpropWeights::learningRateToMultiplier(int batchSize) {
//        float multiplier = rate / batchSize / sqrt(dim.outputSize);
//    float multiplier = rate;
//    std::cout << "rate " << rate << " multiplier " << multiplier << std::endl;
    return 1.0f;
}


