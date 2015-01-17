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

    // [[[end]]]
};

