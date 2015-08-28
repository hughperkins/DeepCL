// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DropoutForward.h"

#define VIRTUAL virtual
#define STATIC static

class CLKernel;

class DropoutForwardGpuNaive : public DropoutForward {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~DropoutForwardGpuNaive();
    VIRTUAL void forward(int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper);
    DropoutForwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, float dropRatio);

    // [[[end]]]
};

