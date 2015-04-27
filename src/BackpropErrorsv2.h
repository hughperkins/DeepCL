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

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT BackpropErrorsv2 {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
//    ActivationFunction const *upstreamFn;

    virtual ~BackpropErrorsv2() {}
    virtual void backward( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput ) = 0;

    // [[[cog
    // import cog_addheaders    
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC BackpropErrorsv2 *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC BackpropErrorsv2 *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropErrorsv2 *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    BackpropErrorsv2( OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL float * backward( int batchSize, float *inputData, float *gradOutput, float *filters );

    // [[[end]]]
};

