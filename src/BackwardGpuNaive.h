#pragma once

#include "Backward.h"
#include "OpenCLHelper.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuNaive : public Backward {
public:
    CLKernel *kernel;
//    CLKernel *broadcastMultiply;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuNaive();
    VIRTUAL void backward( int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper );
    BackwardGpuNaive( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

