// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/Layer.h"
#include "LossLayer.h"
#include "activate/ActivationFunction.h"

#define VIRTUAL virtual
#define STATIC static

class MultinomialCrossEntropy : public LossLayer {
public:

    float *gradOutput;
    int allocatedSize;
    int batchSize;
//    ActivationFunction const*const activationFunction;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    MultinomialCrossEntropy(Layer *previousLayer, MultinomialCrossEntropyMaker const*maker);
    VIRTUAL ~MultinomialCrossEntropy();
    VIRTUAL float*getGradInput();
    VIRTUAL float calcLoss(float const *expected);
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void calcGradInput(float const*expectedOutput);

    // [[[end]]]
};

