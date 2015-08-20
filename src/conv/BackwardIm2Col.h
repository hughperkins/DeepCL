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
    CLWrapper *columnsWrapper;
    int numKernels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    BackwardIm2Col(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackwardIm2Col();
    VIRTUAL void backward(int batchSize,
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);

    // [[[end]]]
};

