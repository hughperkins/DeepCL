#pragma once

#include "Forward.h"

class AddBias;

class Forward3 : public Forward {
public:
    easycl::CLKernel *kernel;
    AddBias *addBias;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward3();
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *dataWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper,
    easycl::CLWrapper *outputWrapper);
    Forward3(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

