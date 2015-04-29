#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsScratchLarge : public BackpropWeights {
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
    VIRTUAL ~BackpropWeightsScratchLarge();
    VIRTUAL void calcGradWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWeightsWrapper );
    BackpropWeightsScratchLarge( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

