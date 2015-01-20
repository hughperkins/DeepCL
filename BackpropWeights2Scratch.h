#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2Scratch : public BackpropWeights2 {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeights2Scratch
    // cppfile: BackpropWeights2Scratch.cpp

    BackpropWeights2Scratch( OpenCLHelper *cl, LayerDimensions dim );
    VIRTUAL ~BackpropWeights2Scratch();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );

    // [[[end]]]
};

