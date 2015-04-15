// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Trainable.h"

#include "DeepCLDllExport.h"

class EpochResult;
class NetAction;

#define VIRTUAL virtual
#define STATIC static

// This class is responsible for running all batches once (ie one 'epoch'), but
// for a single set of already-loaded data
// If you want to have a class that loads the data in chunks, then you'll need
// OnDemandBatcher
// Note however that, what OnDemandBatcher does is, it loads in some data, and
// then it calls this Batcher class.  so they work together
class DeepCL_EXPORT Batcher {
public:
    Trainable *net;
    int batchSize;
    int N;
    float const* data;
    int const* labels;

    int numBatches;
    int inputCubeSize;

    bool epochDone;
    int nextBatch;
    int numRight;
    float loss;

    virtual void internalTick( float const*batchData, int const*batchLabels) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels );
    VIRTUAL ~Batcher();
    void updateVars();
    void reset();
    int getNextBatch();
    bool tick();
    EpochResult run();

    // [[[end]]]
//    Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels );
//    virtual ~Batcher(){}
//    void updateVars();
//    void reset();
//    virtual void internalTick( float const*batchData, int const*batchLabels ) = 0;
//    virtual bool tick();
//    EpochResult run();
//    int getNextBatch();
};


class DeepCL_EXPORT LearnBatcher : public Batcher {
public:
    float learningRate;
    LearnBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, float learningRate );
    virtual void internalTick( float const*batchData, int const*batchLabels);
};


class DeepCL_EXPORT NetActionBatcher : public Batcher {
public:
    NetAction * netAction;
    NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction * netAction);
    virtual void internalTick( float const*batchData, int const*batchLabels );
};


class DeepCL_EXPORT PropagateBatcher : public Batcher {
public:
    PropagateBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels);
    virtual void internalTick( float const*batchData, int const*batchLabels);
};


