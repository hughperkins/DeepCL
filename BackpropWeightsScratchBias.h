#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsScratchBias : public BackpropWeights {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeightsScratchBias
    // cppfile: BackpropWeightsScratchBias.cpp

    BackpropWeightsScratchBias( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    VIRTUAL ~BackpropWeightsScratchBias();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );

    // [[[end]]]
};

