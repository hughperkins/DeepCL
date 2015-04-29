// Copyright Hugh Perkins 2014 hughperkins at gmail
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
class PoolingForward;
class PoolingBackward;
class RandomPatchesMaker;

class RandomPatches : public Layer {
public:
    const int patchSize;
    const int numPlanes;
    const int inputImageSize;

    const int outputImageSize;

    float *output;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    RandomPatches( Layer *previousLayer, RandomPatchesMaker *maker );
    VIRTUAL ~RandomPatches();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getOutputSize();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize() const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL void forward();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

