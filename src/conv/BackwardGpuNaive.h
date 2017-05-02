#pragma once

#include "Backward.h"
#include "EasyCL.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuNaive : public Backward {
public:
    easycl::CLKernel *kernel;
//    CLKernel *broadcastMultiply;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuNaive();
    VIRTUAL void backward(int batchSize,
    easycl::CLWrapper *inputDataWrapper, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *weightsWrapper,
    easycl::CLWrapper *gradInputWrapper);
    BackwardGpuNaive(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

