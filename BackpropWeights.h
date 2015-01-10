#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    ActivationFunction const*fn;

    virtual void backpropWeights( int batchSize, float learningRate, 
        CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeights
    // cppfile: BackpropWeights.cpp

    STATIC BackpropWeights *instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    STATIC BackpropWeights *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    STATIC BackpropWeights *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    BackpropWeights( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    VIRTUAL void backpropWeights( int batchSize, float learningRate, float *errors, float *results, float *inputData, float *filters, float *biasWeights );

    // [[[end]]]
};

