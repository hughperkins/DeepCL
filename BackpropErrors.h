#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrors {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;

    virtual void backpropErrors( int batchSize, 
        CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errors,
        CLWrapper *errorsForUpstream ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropErrors
    // cppfile: BackpropErrors.cpp

    STATIC BackpropErrors *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC BackpropErrors *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropErrors *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    BackpropErrors( OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL float * backpropErrors( int batchSize, float *filters, float *biases, float *errors );

    // [[[end]]]
};

