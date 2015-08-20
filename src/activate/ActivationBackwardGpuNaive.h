// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "activate/ActivationBackward.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationBackwardGpuNaive : public ActivationBackward {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ActivationBackwardGpuNaive();
    VIRTUAL void backward(int batchSize, CLWrapper *inputWrapper,
    CLWrapper *gradOutputWrapper,
    CLWrapper *gradInputWrapper);
    ActivationBackwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn);

    // [[[end]]]
};

