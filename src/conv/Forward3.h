#pragma once

#include "Forward.h"

class AddBias;

class Forward3 : public Forward {
public:
    CLKernel *kernel;
    AddBias *addBias;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward3();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper);
    Forward3(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

