#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsByRow : public BackpropWeights {
public:
    CLKernel *kernel;
    CLKernel *reduce;
    CLKernel *perElementAdd;

    int workgroupSize;
    int numWorkgroups;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsByRow();
    VIRTUAL void backpropWeights(int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper);
    BackpropWeightsByRow(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

