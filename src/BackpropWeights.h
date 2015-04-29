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

class DeepCL_EXPORT BackpropWeights {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    bool debug; // = false;

    virtual ~BackpropWeights() {}
    virtual void calcGradWeights( int batchSize, float learningRate, CLWrapper *gradOutputWrapper, CLWrapper *inputsWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWeightsWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackpropWeights( OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropWeights *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC BackpropWeights *instanceForTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC BackpropWeights *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL void calcGradWeights( int batchSize, float learningRate, float *gradOutput, float *inputs, float *gradWeights, float *gradBiasWeights );
    float learningRateToMultiplier( int batchSize, float rate );

    // [[[end]]]
};

