// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/Layer.h"

class OutputData;

class LossLayer : public Layer {
public:
    virtual float calcLoss(float const*expectedValue) = 0;
    virtual void calcGradInput(float const*expectedOutput) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    LossLayer(Layer *previousLayer, LossLayerMaker *maker);
    VIRTUAL void forward();
    VIRTUAL bool needsBackProp();
    VIRTUAL float *getOutput();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getWeightsSize() const;
    VIRTUAL float calcLoss(OutputData *outputData);
    VIRTUAL void calcGradInput(OutputData *outputData);
    VIRTUAL int calcNumRight(OutputData *outputData);

    // [[[end]]]
};

