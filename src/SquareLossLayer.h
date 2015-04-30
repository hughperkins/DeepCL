// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "LossLayer.h"
#include "ActivationFunction.h"

class SquareLossMaker;

#define VIRTUAL virtual
#define STATIC static

class SquareLossLayer : public LossLayer {
public:

    float *gradInput;
    int allocatedSize;
    int batchSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    SquareLossLayer( Layer *previousLayer, SquareLossMaker *maker );
    VIRTUAL ~SquareLossLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float*getGradInput();
    VIRTUAL float calcLoss( float const *expected );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void calcGradInput( float const*expectedOutput );
    VIRTUAL int getPersistSize() const;
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

