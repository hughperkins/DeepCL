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

class PoolingLayer : public Layer {
public:
    const int numPlanes;
    const int inputBoardSize;
    const int poolingSize;

    OpenCLHelper *const cl; // NOT owned by us
    PoolingPropagate *poolingPropagateImpl;

    float *results;
    float *errorsForUpstream;
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
    // classname: PoolingLayer
    // cppfile: PoolingLayer.cpp

    PoolingLayer( Layer *previousLayer, PoolingMaker const*maker );
    VIRTUAL ~PoolingLayer();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL int getResultsSize();
    VIRTUAL float *getResults();

    // [[[end]]]
};

