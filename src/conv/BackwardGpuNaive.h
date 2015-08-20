#pragma once

#include "Backward.h"
#include "EasyCL.h"

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
    VIRTUAL void backward(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);
    BackwardGpuNaive(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

