#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2ScratchLarge : public BackpropWeights2 {
public:
    CLKernel *kernel;
    int numStripes;
    int inputStripeOuterSize;
    int outputStripeSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeights2ScratchLarge();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );
    BackpropWeights2ScratchLarge( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

