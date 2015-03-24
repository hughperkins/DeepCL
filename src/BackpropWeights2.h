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
#include "ClConvolveDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class ClConvolve_EXPORT BackpropWeights2 {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    bool debug = false;

    virtual ~BackpropWeights2() {}
    virtual void backpropWeights( int batchSize, float learningRate, CLWrapper *derivLossBySumWrapper, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackpropWeights2( OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropWeights2 *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC BackpropWeights2 *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropWeights2 *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL void backpropWeights( int batchSize, float learningRate, float *derivLossBySum, float *inputData, float *filters, float *biasWeights );
    float learningRateToMultiplier( int batchSize, float rate );

    // [[[end]]]
};

