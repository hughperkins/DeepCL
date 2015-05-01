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

#include "RandomSingleton.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "stringhelper.h"
#include "EasyCL.h"

#define VIRTUAL virtual

class TrainerState;
class TrainerStateMaker;

PUBLICAPI
/// A single layer within the neural net
class Layer {
public:
    Layer *previousLayer;
    Layer *nextLayer;
    const int layerIndex;
    bool training;

    LayerMaker2 *maker;

    // \brief Get the activated output from this layer, after forward propagation
    PUBLICAPI virtual float * getOutput() = 0;
//    virtual Layer *clone() = 0;
    /// \brief Get the size of array needed for persisting to/from an array
    PUBLICAPI virtual int getPersistSize() const = 0;
    /// \brief Get the size of the activated output from this layer
    PUBLICAPI virtual int getOutputSize() const = 0;
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
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL float *getGradInput();
    VIRTUAL CLWrapper *getGradWeightsWrapper();
    VIRTUAL CLWrapper *getGradBiasWrapper();
    VIRTUAL CLWrapper *getWeightsWrapper();
    VIRTUAL CLWrapper *getBiasWrapper();
    VIRTUAL CLWrapper *getGradInputWrapper();
    PUBLICAPI VIRTUAL bool getBiased() const;
    PUBLICAPI VIRTUAL bool hasOutputWrapper() const;
    PUBLICAPI VIRTUAL CLWrapper *getOutputWrapper();
    PUBLICAPI VIRTUAL int getOutputCubeSize() const;
    PUBLICAPI VIRTUAL int getOutputPlanes() const;
    PUBLICAPI VIRTUAL int getOutputImageSize() const;
    VIRTUAL void forward();
    VIRTUAL bool needsBackProp();
    VIRTUAL void print();
    VIRTUAL void initWeights( float const*weights );
    VIRTUAL void initBias( float const *bias );
    VIRTUAL void printWeightsAsCode() const;
    VIRTUAL void printBiasAsCode() const;
    VIRTUAL void printWeights();
    VIRTUAL void printOutput();
    PUBLICAPI VIRTUAL void backward();
    VIRTUAL float *getGradWeights();
    VIRTUAL float *getGradBias();
    VIRTUAL bool biased();
    PUBLICAPI VIRTUAL int getWeightsSize() const;
    PUBLICAPI VIRTUAL int getBiasSize() const;
    PUBLICAPI VIRTUAL void persistToArray(float *array);
    PUBLICAPI VIRTUAL void unpersistFromArray(float const*array);
    VIRTUAL void setWeights(float *weights, float *bias);
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL float const*getBias() const;
    PUBLICAPI VIRTUAL std::string asString() const;
    VIRTUAL bool needsTrainerState  () const;
    VIRTUAL void setTrainerState( TrainerStateMaker *trainerMaker );
    VIRTUAL TrainerState *getTrainerState();
    VIRTUAL TrainerState *getBiasTrainerState();
    VIRTUAL void updateWeights( CLWrapper *weightChangesWrapper, CLWrapper *biasChangesWrapper );

    // [[[end]]]

};

