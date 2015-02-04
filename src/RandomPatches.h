// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#define VIRTUAL virtual
#define STATIC static

class CLKernel;
class CLWrapper;
class PoolingPropagate;
class PoolingBackprop;

class RandomPatches : public Layer {
public:
    const int patchSize;
    const int numPlanes;
    const int inputBoardSize;

    const int outputBoardSize;

    float *results;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    RandomPatches( Layer *previousLayer, RandomPatchesMaker const *maker );
    VIRTUAL ~RandomPatches();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getResultsSize();
    VIRTUAL float *getResults();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getResultsSize() const;
    VIRTUAL int getOutputBoardSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize() const;
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL void propagate();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

