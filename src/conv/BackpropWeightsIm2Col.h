#pragma once

#include "BackpropWeights.h"
#include "EasyCL.h"

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT BackpropWeightsIm2Col : public BackpropWeights {
    private:
    CLKernel *kernelIm2Col;

    float *columns;
    CLWrapper *columnsWrapper;
    int numKernels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    BackpropWeightsIm2Col(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackpropWeightsIm2Col();
    VIRTUAL void calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *inputWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper);

    private:
    STATIC std::string getKernelTemplate();

    // [[[end]]]
};

