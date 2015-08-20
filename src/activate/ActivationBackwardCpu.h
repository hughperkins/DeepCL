// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "activate/ActivationBackward.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationBackwardCpu : public ActivationBackward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationBackwardCpu(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn);
    VIRTUAL void backward(int batchSize, float *outputs, float *gradOutput, float *gradInput);
    VIRTUAL void backward(int batchSize,
    CLWrapper *outputWrapper,
    CLWrapper *gradOutputWrapper,
    CLWrapper *gradInputWrapper);

    // [[[end]]]
};

