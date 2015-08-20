// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/Layer.h"

#define VIRTUAL virtual
#define STATIC static

class CLKernel;
class CLWrapper;
class DropoutForward;
class DropoutBackward;
class RandomSingleton;
class DropoutMaker;
class MultiplyBuffer;

class DropoutLayer : public Layer {
public:
    const int numPlanes;
    const int inputSize;
    const float dropRatio;

    const int outputSize;

    RandomSingleton *random;

    EasyCL *const cl; // NOT owned by us
    DropoutForward *dropoutForwardImpl;
    DropoutBackward *dropoutBackwardImpl;
    MultiplyBuffer *multiplyBuffer; // for skipping dropout...

    unsigned char *masks;
    float *output;
    float *gradInput;

    CLWrapper *maskWrapper;
    CLWrapper *outputWrapper;
    CLWrapper *gradInputWrapper;

//    bool outputCopiedToHost;
//    bool gradInputCopiedToHost;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutLayer(EasyCL *cl, Layer *previousLayer, DropoutMaker *maker);
    VIRTUAL ~DropoutLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void fortesting_setRandomSingleton(RandomSingleton *random);
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getOutputNumElements();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL float *getGradInput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void generateMasks();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

