// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ConvolutionalLayer.h"

#define VIRTUAL virtual
#define STATIC static

class FullyConnectedLayer : public Layer {
public:
    const int numPlanes;
    const int boardSize;
    ActivationFunction const*fn;

    ConvolutionalLayer *convolutionalLayer;
    int batchSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    FullyConnectedLayer( OpenCLHelper *cl, Layer *previousLayer, FullyConnectedMaker const*maker );
    VIRTUAL ~FullyConnectedLayer();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getOutputBoardSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize() const;
    VIRTUAL void persistToArray(float *array);
    VIRTUAL void unpersistFromArray(float const*array);
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasWeightsSize() const;
    VIRTUAL int getResultsSize() const;
    VIRTUAL float *getResults();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL CLWrapper *getResultsWrapper();
    VIRTUAL ActivationFunction const*getActivationFunction();
    VIRTUAL bool needsBackProp();
    VIRTUAL void propagate();
    VIRTUAL void backProp( float learningRate );
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

