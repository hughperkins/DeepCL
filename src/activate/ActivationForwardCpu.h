// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "activate/ActivationForward.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationForwardCpu : public ActivationForward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationForwardCpu(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn);
    VIRTUAL void forward(int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper);
    VIRTUAL void forward(int batchSize, float *input, float *output);

    // [[[end]]]
};

