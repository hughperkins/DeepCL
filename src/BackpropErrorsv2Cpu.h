// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "BackpropErrorsv2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrorsv2Cpu : public BackpropErrorsv2 {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackpropErrorsv2Cpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn );
    VIRTUAL ~BackpropErrorsv2Cpu();
    VIRTUAL float *backpropErrors( int batchSize, float *inputData,
    float *errors, float *weights );
    VIRTUAL void backpropErrors( int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper );

    // [[[end]]]
};

