#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsScratchLarge : public BackpropWeights {
public:
    easycl::CLKernel *kernel;
    int numStripes;
    int inputStripeOuterSize;
    int outputStripeSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsScratchLarge();
    VIRTUAL void calcGradWeights(int batchSize, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *imagesWrapper, easycl::CLWrapper *gradWeightsWrapper, easycl::CLWrapper *gradBiasWrapper);
    BackpropWeightsScratchLarge(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

