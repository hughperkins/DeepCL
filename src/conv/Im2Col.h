#pragma once

#include "LayerDimensions.h"

namespace easycl {
class EasyCL;
class CLWrapper;
class CLKernel;
class TemplatedKernel;
}

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class Im2Col {
    easycl::EasyCL *cl;
    LayerDimensions dim;

    easycl::CLKernel *kernelIm2Col;
    easycl::CLKernel *kernelCol2Im;

    int numKernelsIm2Col;
    int numKernelsCol2Im;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    Im2Col(easycl::EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~Im2Col();
    void im2Col(easycl::CLWrapper *imagesWrapper, int imagesOffset, easycl::CLWrapper *columnsWrapper);
    void col2Im(easycl::CLWrapper *columnsWrapper, easycl::CLWrapper *imagesWrapper, int imagesOffset);

    private:
    void setupBuilder(easycl::TemplatedKernel *builder);
    void buildKernelIm2Col();
    void buildKernelCol2Im();
    STATIC std::string getKernelTemplate();

    // [[[end]]]
};

