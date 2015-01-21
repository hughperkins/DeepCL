#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2Naive : public BackpropWeights2 {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeights2Naive
    // cppfile: BackpropWeights2Naive.cpp

    BackpropWeights2Naive( OpenCLHelper *cl, LayerDimensions dim );
    VIRTUAL ~BackpropWeights2Naive();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );

    // [[[end]]]
};

