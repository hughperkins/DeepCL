// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Forward.h"

class AddBias;
class Im2Col;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT ForwardIm2Col : public Forward {
    private:
//    CLKernel *kernelIm2Col;
//    CLKernel *kernelCol2Im;
    AddBias *addBias;
    Im2Col *im2Col;

    float *columns;
    CLWrapper *columnsWrapper;
    int numKernels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    ForwardIm2Col(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~ForwardIm2Col();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper);

    // [[[end]]]
};

