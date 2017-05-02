#pragma once

#include "conv/Forward.h"

class AddBias;

class Forward4 : public Forward {
public:
    easycl::CLKernel *kernel;
    AddBias *addBias;

    int workgroupSize;
    int pixelsPerThread;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward4();
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *dataWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper,
    easycl::CLWrapper *outputWrapper);
    Forward4(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

