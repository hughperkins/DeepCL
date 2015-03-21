// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "MyRandom.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "stringhelper.h"
#include "OpenCLHelper.h"

#define VIRTUAL virtual

class Layer {
public:
    Layer *previousLayer;
    Layer *nextLayer;
    const int layerIndex;
    bool training;

    LayerMaker2 *maker;

    virtual float * getResults() = 0;
//    virtual Layer *clone() = 0;
    virtual int getPersistSize() const = 0;
    virtual int getResultsSize() const = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Layer( Layer *previousLayer, LayerMaker2 *maker );
    VIRTUAL ~Layer();
    VIRTUAL void setTraining( bool training );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    VIRTUAL bool getBiased() const;
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL CLWrapper *getResultsWrapper();
    VIRTUAL ActivationFunction const*getActivationFunction();
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL void propagate();
    VIRTUAL bool needsBackProp();
    VIRTUAL void print();
    VIRTUAL void initWeights( float const*weights );
    VIRTUAL void initBiasWeights( float const *biasWeights );
    VIRTUAL void printWeightsAsCode() const;
    VIRTUAL void printBiasWeightsAsCode() const;
    VIRTUAL void printWeights();
    VIRTUAL void printOutput() const;
    VIRTUAL void backProp( float learningRate );
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasWeightsSize() const;
    VIRTUAL void persistToArray(float *array);
    VIRTUAL void unpersistFromArray(float const*array);
    VIRTUAL void setWeights(float *weights, float *biasWeights);
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL float const*getBiasWeights() const;
    VIRTUAL std::string asString() const;

    // [[[end]]]

};

std::ostream &operator<<(std::ostream&os, Layer const*layer );

