// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "LossLayer.h"

#define VIRTUAL virtual
#define STATIC static

// this doesnt have any weights as such, just handles propagation, and backpropagation
// it will have the same shape as the previous layer, ie same boardsize, same number of planes
// the softmax will be per-plane, or maybe that is configurable?
// this will ALWAYS use multinomial logistic loss (ie cross-entropy loss), at least for now
class SoftMaxLayer : public LossLayer {
public:
    const bool perPlane;
    const int boardSize;
    const int numPlanes;

    float *results;
    float *errorsForUpstream;
    int allocatedSize;
    int batchSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: SoftMaxLayer
    // cppfile: SoftMaxLayer.cpp

    SoftMaxLayer(  Layer *previousLayer, SoftMaxMaker const *maker  );
    VIRTUAL ~SoftMaxLayer();
    VIRTUAL float *getResults();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL float calcLoss( float const *expectedValues );
    VIRTUAL void calcErrors( float const *expectedValues );
    VIRTUAL void propagate();
    VIRTUAL void backPropErrors( float learningRate );

    // [[[end]]]
};

