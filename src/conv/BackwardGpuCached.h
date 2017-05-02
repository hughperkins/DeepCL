#pragma once

#include "Backward.h"
#include "EasyCL.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuCached : public Backward {
public:
    easycl::CLKernel *kernel;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuCached();
    VIRTUAL void backward(int batchSize,
    easycl::CLWrapper *inputDataWrapper, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *weightsWrapper,
    easycl::CLWrapper *gradInputWrapper);
    BackwardGpuCached(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

