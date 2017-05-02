#pragma once

#include "Forward.h"

class ForwardByInputPlane : public Forward {
public:
    easycl::CLKernel *kernel;
    easycl::CLKernel *reduceSegments;
    easycl::CLKernel *repeatedAdd;
//    CLKernel *activate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardByInputPlane();
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *dataWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper,
    easycl::CLWrapper *outputWrapper);
    ForwardByInputPlane(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

