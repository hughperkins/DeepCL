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
    // generated, using cog:
    VIRTUAL ~BackpropWeights2Scratch();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );
    BackpropWeights2Scratch( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

