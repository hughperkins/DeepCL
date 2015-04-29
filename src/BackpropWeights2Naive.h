// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2Naive : public BackpropWeights2 {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeights2Naive();
    VIRTUAL void calcGradWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWeightsWrapper );
    BackpropWeights2Naive( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

