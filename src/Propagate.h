#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class Propagate {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    ActivationFunction const*fn;

    virtual void propagate( int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
        CLWrapper *resultsWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: Propagate
    // cppfile: Propagate.cpp

    STATIC Propagate *instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    STATIC Propagate *instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    STATIC Propagate *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const*fn );
    VIRTUAL float * propagate( int batchSize, float *inputData, float *filters, float *biases );

    // [[[end]]]

};



