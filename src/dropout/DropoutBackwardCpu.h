// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DropoutBackward.h"

#define VIRTUAL virtual
#define STATIC static

class DropoutBackwardCpu : public DropoutBackward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutBackwardCpu(EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    VIRTUAL void backward(int batchSize, uchar *mask,  float *gradOutput, float *gradInput);
    VIRTUAL void backward(int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper,
    CLWrapper *gradInputWrapper);

    // [[[end]]]
};

