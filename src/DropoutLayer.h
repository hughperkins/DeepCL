// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"

#define VIRTUAL virtual
#define STATIC static

class CLKernel;
class CLWrapper;
class DropoutPropagate;
class DropoutBackprop;
class RandomSingleton;
class DropoutMaker;
class MultiplyBuffer;

class DropoutLayer : public Layer {
public:
    const int numPlanes;
    const int inputImageSize;
    const float dropRatio;

    const int outputImageSize;

    RandomSingleton *random;

    OpenCLHelper *const cl; // NOT owned by us
    DropoutPropagate *dropoutPropagateImpl;
    DropoutBackprop *dropoutBackpropImpl;
    MultiplyBuffer *multiplyBuffer; // for skipping dropout...

    unsigned char *masks;
    float *results;
    float *errorsForUpstream;

    CLWrapper *maskWrapper;
    CLWrapper *resultsWrapper;
    CLWrapper *errorsForUpstreamWrapper;

    bool resultsCopiedToHost;
    bool errorsForUpstreamCopiedToHost;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutLayer( OpenCLHelper *cl, Layer *previousLayer, DropoutMaker *maker );
    VIRTUAL ~DropoutLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void fortesting_setRandomSingleton( RandomSingleton *random );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getResultsSize();
    VIRTUAL float *getResults();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getResultsSize() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize() const;
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL CLWrapper *getResultsWrapper();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void generateMasks();
    VIRTUAL void propagate();
    VIRTUAL void backProp( float learningRate );
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

