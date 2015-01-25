// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrorsv2 {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    ActivationFunction const *upstreamFn;

    virtual void backpropErrors( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *errors, CLWrapper *weightsWrapper,
        CLWrapper *errorsForUpstream ) = 0;

    // [[[cog
    // import cog_addheaders    
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC BackpropErrorsv2 *instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn );
    STATIC BackpropErrorsv2 *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn );
    STATIC BackpropErrorsv2 *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn );
    BackpropErrorsv2( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *upstreamFn );
    VIRTUAL float * backpropErrors( int batchSize, float *inputData, float *errors, float *filters );

    // [[[end]]]
};

