// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "BackpropWeights.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsCpu : public BackpropWeights {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BackpropWeightsCpu(easycl::EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~BackpropWeightsCpu();
    VIRTUAL void calcGradWeights(int batchSize, easycl::CLWrapper *gradOutputWrapper, easycl::CLWrapper *imagesWrapper, easycl::CLWrapper *gradWeightsWrapper, easycl::CLWrapper *gradBiasWrapper);
    VIRTUAL void calcGradWeights(int batchSize, float *gradOutput,
    float *inputs, float *gradWeights, float *gradBias);

    // [[[end]]]
};

