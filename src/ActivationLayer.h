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

class ActivationPropagate;
class ActivationBackprop;
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

    OpenCLHelper *const cl; // NOT owned by us
    ActivationPropagate *activationPropagateImpl;
    ActivationBackprop *activationBackpropImpl;

    float *output;
    float *errorsForUpstream;

    CLWrapper *outputWrapper;
    CLWrapper *errorsForUpstreamWrapper;

    bool outputCopiedToHost;
    bool errorsForUpstreamCopiedToHost;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationLayer( OpenCLHelper *cl, Layer *previousLayer, ActivationMaker *maker );
    VIRTUAL ~ActivationLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getOutputSize();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL float *getGradInput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void propagate();
    VIRTUAL void backProp( float learningRate );
    VIRTUAL std::string asString() const;
    VIRTUAL int getPersistSize() const;

    // [[[end]]]
};

