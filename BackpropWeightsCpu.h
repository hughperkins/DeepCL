#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsCpu : public BackpropWeights {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeightsCpu
    // cppfile: BackpropWeightsCpu.cpp

    BackpropWeightsCpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    VIRTUAL ~BackpropWeightsCpu();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );
    VIRTUAL void backpropWeights( int batchSize, float learningRate, float *errors,
    float *results, float *images, float *weights, float *biasWeights );

    // [[[end]]]
};

