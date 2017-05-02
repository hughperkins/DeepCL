// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "PoolingBackward.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingBackwardCpu : public PoolingBackward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PoolingBackwardCpu(easycl::EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize);
    VIRTUAL void backward(int batchSize,  float *gradOutput, int *selectors, float *gradInput);
    VIRTUAL void backward(int batchSize, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *selectorsWrapper,
    easycl::CLWrapper *gradInputWrapper);

    // [[[end]]]
};

