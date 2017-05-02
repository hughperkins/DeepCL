#pragma once

#include "Backward.h"
#include "EasyCL.h"

#include "DeepCLDllExport.h"

class Im2Col;

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT BackwardIm2Col : public Backward {
    private:
    Im2Col *im2Col;
//    CLKernel *kernelCol2Im;
//    AddBias *addBias;

    float *columns;
    easycl::CLWrapper *columnsWrapper;
    int numKernels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    BackwardIm2Col(easycl::EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackwardIm2Col();
    VIRTUAL void backward(int batchSize,
        easycl::CLWrapper *inputDataWrapper, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *weightsWrapper,
    easycl::CLWrapper *gradInputWrapper);

    // [[[end]]]
};

