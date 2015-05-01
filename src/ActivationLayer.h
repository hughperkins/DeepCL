// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "ActivationFunction.h"
#include "Layer.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationForward;
class ActivationBackward;
class ActivationMaker;

// this will contain only activation, and then we can factorize activations away from
// the convolutional layers etc
// thats the plan :-)
class ActivationLayer : public Layer {
public:
    const int numPlanes;
    const int inputImageSize;

    const int outputImageSize;

    ActivationFunction const *fn;

    EasyCL *const cl; // NOT owned by us
    ActivationForward *activationForwardImpl;
    ActivationBackward *activationBackpropImpl;

    float *output; // this is not guaranteed to be up to date
                // unless outputCopiedToHost is true
    float *gradInput; // this is not guaranteed to be up to date
                // unless gradInputCopiedToHost is true

    CLWrapper *outputWrapper; // this is guaranteed to be up to date
    CLWrapper *gradInputWrapper; // this is guaranteed to be up to date

    bool outputCopiedToHost;
    bool gradInputCopiedToHost;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationLayer( EasyCL *cl, Layer *previousLayer, ActivationMaker *maker );
    VIRTUAL ~ActivationLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float getOutput( int n, int plane, int row, int col );
    VIRTUAL void printOutput();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getOutputSize();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasSize() const;
    VIRTUAL float *getGradInput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL std::string asString() const;
    VIRTUAL int getPersistSize() const;

    // [[[end]]]
};

