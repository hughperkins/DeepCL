#pragma once

#include "Layer.h"

#define VIRTUAL virtual
#define STATIC static

// this doesnt have any weights as such, just handles propagation, and backpropagation
// it will have the same shape as the previous layer, ie same boardsize, same number of planes
// the softmax will be per-plane, or maybe that is configurable?
class SoftMaxLayer : public Layer {
public:
    float *errorsForUpstream;
    bool allocatedSize;

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
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL void propagate();
    VIRTUAL void backPropErrors( float learningRate );

    // [[[end]]]
};

