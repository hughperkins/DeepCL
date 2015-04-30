#pragma once

#include "Backward.h"
#include "OpenCLHelper.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuCached : public Backward {
public:
    CLKernel *kernel;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuCached();
    VIRTUAL void backward( int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper );
    BackwardGpuCached( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

