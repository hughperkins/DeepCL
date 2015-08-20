// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Backward.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardCpu : public Backward {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackwardCpu(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackwardCpu();
    VIRTUAL float *backward(int batchSize, float *inputs,
    float *gradOutput, float *weights);
    VIRTUAL void backward(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);

    // [[[end]]]
};

