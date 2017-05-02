// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"

class FullyConnectedMaker;

#define VIRTUAL virtual
#define STATIC static

class FullyConnectedLayer : public Layer {
public:
    const int numPlanes;
    const int imageSize;
//    ActivationFunction const*fn;

    ConvolutionalLayer *convolutionalLayer;
    int batchSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    FullyConnectedLayer(easycl::EasyCL *cl, Layer *previousLayer, FullyConnectedMaker *maker);
    VIRTUAL ~FullyConnectedLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL void persistToArray(int version, float *array);
    VIRTUAL void unpersistFromArray(int version, float const*array);
    VIRTUAL void setWeights(float *weights, float *bias);
    VIRTUAL float * getWeights();
    VIRTUAL float * getBias();
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasSize() const;
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL float *getOutput();
    VIRTUAL float *getGradInput();
    VIRTUAL easycl::CLWrapper *getGradWeightsWrapper();
    VIRTUAL easycl::CLWrapper *getGradBiasWrapper();
    VIRTUAL easycl::CLWrapper *getWeightsWrapper();
    VIRTUAL easycl::CLWrapper *getBiasWrapper();
    VIRTUAL bool biased();
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL easycl::CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL easycl::CLWrapper *getOutputWrapper();
    VIRTUAL bool needsBackProp();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL bool needsTrainerState() const;
    VIRTUAL TrainerState *getTrainerState();
    VIRTUAL TrainerState *getBiasTrainerState();
    VIRTUAL void setTrainerState(TrainerStateMaker *TrainerStateMaker);
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

