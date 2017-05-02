// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "PoolingForward.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingForwardCpu : public PoolingForward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PoolingForwardCpu(easycl::EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize);
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *inputWrapper, easycl::CLWrapper *selectorsWrapper, easycl::CLWrapper *outputWrapper);
    VIRTUAL void forward(int batchSize, float *input, int *selectors, float *output);

    // [[[end]]]
};

