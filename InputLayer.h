// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"

class InputLayer : public Layer {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: InputLayer
    // cppfile: InputLayer.cpp

    InputLayer( Layer *previousLayer, InputLayerMaker const*maker );
    virtual float *getResults();
    virtual void printOutput() const;
    virtual void print() const;
    void in( float const*images );
    virtual ~InputLayer();
    virtual bool needErrorsBackprop();
    virtual void setBatchSize( int batchSize );
    virtual void propagate();
    virtual void backPropErrors( float learningRate, float const *errors );

    // [[[end]]]
};

