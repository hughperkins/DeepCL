#pragma once

#include "BackpropErrors.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrorsCpu : public BackpropErrors {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropErrorsCpu
    // cppfile: BackpropErrorsCpu.cpp

    BackpropErrorsCpu( OpenCLHelper *cl, LayerDimensions dim );
    VIRTUAL ~BackpropErrorsCpu();
    VIRTUAL float *backpropErrors( int batchSize, float *weights, float *biasWeights,
    float *errors );
    VIRTUAL void backpropErrors( int batchSize,
    CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
    CLWrapper *errorsForUpstreamWrapper );

    // [[[end]]]
};

