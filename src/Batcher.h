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
class Trainer;

#define VIRTUAL virtual
#define STATIC static

/// \brief Runs an epoch for a single set of already-loaded data
///
/// This class is responsible for running all batches once (ie one 'epoch'), but
/// for a single set of already-loaded data
/// If you want to have a class that loads the data in chunks, then you'll need
/// OnDemandBatcher
/// Note however that, what OnDemandBatcher does is, it loads in some data, and
/// then it calls this Batcher class.  so they work together
PUBLICAPI
class DeepCL_EXPORT Batcher {
protected:
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

public:
    virtual void internalTick( float const*batchData, int const*batchLabels) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels );
    VIRTUAL ~Batcher();
    PUBLICAPI void reset();
    PUBLICAPI int getNextBatch();
    PUBLICAPI VIRTUAL float getLoss();
    PUBLICAPI VIRTUAL int getNumRight();
    PUBLICAPI VIRTUAL int getN();
    PUBLICAPI VIRTUAL bool getEpochDone();
    VIRTUAL void setBatchState( int nextBatch, int numRight, float loss );
    VIRTUAL void setN( int N );
    PUBLICAPI bool tick();
    PUBLICAPI EpochResult run();

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
    Trainer *trainer; // NOT delete

//    float learningRate;
//    virtual void setLearningRate( float learningRate ) {
//        this->learningRate = learningRate;
//    }
    LearnBatcher( Trainer *trainer, Trainable *net, int batchSize, int N, float *data, int const*labels );
    virtual void internalTick( float const*batchData, int const*batchLabels);
//    float getLearningRate() {
//        return learningRate;
//    }
};

class DeepCL_EXPORT NetActionBatcher : public Batcher {
public:
    NetAction * netAction;
    NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction * netAction);
    virtual void internalTick( float const*batchData, int const*batchLabels );
};


class DeepCL_EXPORT ForwardBatcher : public Batcher {
public:
    ForwardBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels);
    virtual void internalTick( float const*batchData, int const*batchLabels);
};


