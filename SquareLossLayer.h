#pragma once

#include "Layer.h"
#include "LossLayer.h"
#include "ActivationFunction.h"

#define VIRTUAL virtual
#define STATIC static

class SquareLossLayer : public LossLayer {
public:

    float *derivLossBySum;
    int allocatedSize;
    int batchSize;
//    ActivationFunction const*const activationFunction;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: SquareLossLayer
    // cppfile: SquareLossLayer.cpp

    SquareLossLayer( Layer *previousLayer, SquareLossMaker const*maker );
    VIRTUAL ~SquareLossLayer();
    VIRTUAL float*getDerivLossBySum();
    VIRTUAL float calcLoss( float const *expected );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void calcDerivLossBySum( float const*expectedResults );

    // [[[end]]]
};

