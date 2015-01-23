#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsNaive : public BackpropWeights {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeightsNaive
    // cppfile: BackpropWeightsNaive.cpp

    BackpropWeightsNaive( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    VIRTUAL ~BackpropWeightsNaive();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );

    // [[[end]]]
};

