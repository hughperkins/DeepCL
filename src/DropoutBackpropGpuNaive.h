// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DropoutBackprop.h"

#define VIRTUAL virtual
#define STATIC static

class DropoutBackpropGpuNaive : public DropoutBackprop {
public:
    CLKernel *kernel;
//    CLKernel *kMemset;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~DropoutBackpropGpuNaive();
    VIRTUAL void backpropErrors(
    int batchSize,
    CLWrapper *maskWrapper,
    CLWrapper *errorsWrapper,
    CLWrapper *gradInputWrapper );
    DropoutBackpropGpuNaive( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio );

    // [[[end]]]
};

