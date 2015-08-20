#pragma once

#include "Forward.h"

class ForwardByInputPlane : public Forward {
public:
    CLKernel *kernel;
    CLKernel *reduceSegments;
    CLKernel *repeatedAdd;
//    CLKernel *activate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardByInputPlane();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper);
    ForwardByInputPlane(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

