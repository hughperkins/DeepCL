// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActivationBackprop.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationBackpropGpuNaive : public ActivationBackprop {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ActivationBackpropGpuNaive();
    VIRTUAL void backpropErrors( int batchSize, CLWrapper *inputWrapper,
    CLWrapper *errorsWrapper,
    CLWrapper *gradInputWrapper );
    ActivationBackpropGpuNaive( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );

    // [[[end]]]
};

