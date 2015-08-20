// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Forward.h"

#define STATIC static
#define VIRTUAL virtual

class ForwardCpu : public Forward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ForwardCpu(EasyCL *cl, LayerDimensions dim);
    VIRTUAL void forward(int batchSize, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper);
    VIRTUAL float *forward(int batchSize, float *inputData, float *weights, float *bias);

    // [[[end]]]
};

