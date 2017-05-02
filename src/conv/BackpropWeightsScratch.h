#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsScratch : public BackpropWeights {
public:
    easycl::CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsScratch();
    VIRTUAL void calcGradWeights(int batchSize, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *imagesWrapper, easycl::CLWrapper *gradWeightsWrapper, easycl::CLWrapper *gradBiasWrapper);
    BackpropWeightsScratch(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

