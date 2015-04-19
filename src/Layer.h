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

PUBLICAPI
/// A single layer within the neural net
class Layer {
public:
    Layer *previousLayer;
    Layer *nextLayer;
    const int layerIndex;
    bool training;

    LayerMaker2 *maker;

    virtual float * getResults() = 0;
//    virtual Layer *clone() = 0;
    /// \brief Get the size of array needed for persisting to/from an array
    PUBLICAPI virtual int getPersistSize() const = 0;
    virtual int getResultsSize() const = 0;
    virtual std::string getClassName() const = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI Layer( Layer *previousLayer, LayerMaker2 *maker );
    VIRTUAL ~Layer();
    PUBLICAPI VIRTUAL void setTraining( bool training );
    PUBLICAPI VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    PUBLICAPI VIRTUAL bool getBiased() const;
    PUBLICAPI VIRTUAL bool hasResultsWrapper() const;
    PUBLICAPI VIRTUAL CLWrapper *getResultsWrapper();
    PUBLICAPI VIRTUAL ActivationFunction const*getActivationFunction();
    PUBLICAPI VIRTUAL int getOutputCubeSize() const;
    PUBLICAPI VIRTUAL int getOutputPlanes() const;
    PUBLICAPI VIRTUAL int getOutputImageSize() const;
    PUBLICAPI VIRTUAL void propagate();
    PUBLICAPI VIRTUAL bool needsBackProp();
    VIRTUAL void print();
    PUBLICAPI VIRTUAL void initWeights( float const*weights );
    PUBLICAPI VIRTUAL void initBiasWeights( float const *biasWeights );
    VIRTUAL void printWeightsAsCode() const;
    VIRTUAL void printBiasWeightsAsCode() const;
    VIRTUAL void printWeights();
    VIRTUAL void printOutput() const;
    PUBLICAPI VIRTUAL void backProp( float learningRate );
    PUBLICAPI VIRTUAL int getWeightsSize() const;
    PUBLICAPI VIRTUAL int getBiasWeightsSize() const;
    PUBLICAPI VIRTUAL void persistToArray(float *array);
    PUBLICAPI VIRTUAL void unpersistFromArray(float const*array);
    VIRTUAL void setWeights(float *weights, float *biasWeights);
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL float const*getBiasWeights() const;
    PUBLICAPI VIRTUAL std::string asString() const;

    // [[[end]]]

};

std::ostream &operator<<(std::ostream&os, Layer const*layer );

