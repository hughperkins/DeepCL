#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsByRow : public BackpropWeights {
public:
    easycl::CLKernel *kernel;
    easycl::CLKernel *reduce;
    easycl::CLKernel *perElementAdd;

    int workgroupSize;
    int numWorkgroups;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsByRow();
    VIRTUAL void backpropWeights(int batchSize, float learningRate,  easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *imagesWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper);
    BackpropWeightsByRow(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

