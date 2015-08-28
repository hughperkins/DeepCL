#pragma once

#include "LayerDimensions.h"

class EasyCL;
class CLWrapper;
class CLKernel;
class TemplatedKernel;

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class Im2Col {
    EasyCL *cl;
    LayerDimensions dim;

    CLKernel *kernelIm2Col;
    CLKernel *kernelCol2Im;

    int numKernelsIm2Col;
    int numKernelsCol2Im;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    Im2Col(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~Im2Col();
    void im2Col(CLWrapper *imagesWrapper, int imagesOffset, CLWrapper *columnsWrapper);
    void col2Im(CLWrapper *columnsWrapper, CLWrapper *imagesWrapper, int imagesOffset);

    private:
    void setupBuilder(TemplatedKernel *builder);
    void buildKernelIm2Col();
    void buildKernelCol2Im();
    STATIC std::string getKernelTemplate();

    // [[[end]]]
};

