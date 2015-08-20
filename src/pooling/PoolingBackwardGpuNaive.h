// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "PoolingBackward.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingBackwardGpuNaive : public PoolingBackward {
public:
    CLKernel *kernel;
    CLKernel *kMemset;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PoolingBackwardGpuNaive();
    VIRTUAL void backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper,
    CLWrapper *gradInputWrapper);
    PoolingBackwardGpuNaive(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize);

    // [[[end]]]
};

