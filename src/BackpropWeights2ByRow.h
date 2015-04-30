#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2ByRow : public BackpropWeights2 {
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
    VIRTUAL ~BackpropWeights2ByRow();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );
    BackpropWeights2ByRow( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

