#pragma once

#include "conv/Forward.h"

class AddBias;

class Forward4 : public Forward {
public:
    CLKernel *kernel;
    AddBias *addBias;

    int workgroupSize;
    int pixelsPerThread;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward4();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper);
    Forward4(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

