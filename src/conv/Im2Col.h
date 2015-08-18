#pragma once

#include "LayerDimensions.h"

class EasyCL;
class CLWrapper;

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class Im2Col {
    EasyCL *cl;
    LayerDimensions dim;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    Im2Col(EasyCL *cl, LayerDimensions dim);
    void im2Col(CLWrapper *im, int64 im_offset, CLWrapper *columns);
    void col2Im(CLWrapper *columns, CLWrapper *im, int64 im_offset);

    // [[[end]]]
};

