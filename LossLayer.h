// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"

class LossLayer : public Layer {
public:
    virtual float calcLoss( float const*expectedValue ) = 0;
    virtual void calcDerivLossBySum( float const*expectedResults ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: LossLayer
    // cppfile: LossLayer.cpp

    LossLayer( Layer *previousLayer, LayerMaker const*maker );
    VIRTUAL void propagate();
    VIRTUAL float *getResults();
    VIRTUAL int getOutputCubeSize() const;

    // [[[end]]]
};

