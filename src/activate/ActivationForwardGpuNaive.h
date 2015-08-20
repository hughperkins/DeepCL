// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "activate/ActivationForward.h"

#define VIRTUAL virtual
#define STATIC static

class CLKernel;

class ActivationForwardGpuNaive : public ActivationForward {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ActivationForwardGpuNaive();
    VIRTUAL void forward(int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper);
    ActivationForwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn);

    // [[[end]]]
};

