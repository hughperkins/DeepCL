// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DropoutForward.h"

#define VIRTUAL virtual
#define STATIC static

class DropoutForwardCpu : public DropoutForward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutForwardCpu(easycl::EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *masksWrapper, easycl::CLWrapper *inputWrapper, easycl::CLWrapper *outputWrapper);
    VIRTUAL void forward(int batchSize, unsigned char *masks, float *input, float *output);

    // [[[end]]]
};

