// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Trainable.h"

// class NeuralNet;
//class Trainable;

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

class DeepCL_EXPORT EpochResult {
public:
    float loss;
    int numRight;
    EpochResult( float loss, int numRight ) :
        loss( loss ),
        numRight( numRight ) {
    }
};

class DeepCL_EXPORT PostBatchAction {
public:
    virtual void run( int batch, int numRightSoFar, float lossSoFar ) = 0;
};


class DeepCL_EXPORT NetAction {
public:
    virtual ~NetAction() {}
    virtual void run( Trainable *net, float const*const batchData, int const*const batchLabels ) = 0;
};


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

    std::vector< PostBatchAction * > postBatchActions; // note: we DONT own these, dont delete, caller owns

    Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels );
    virtual ~Batcher(){}
    void updateVars();
    void reset();
    virtual void _tick( float const*batchData, int const*batchLabels ) {
        throw std::runtime_error("batcher::_tick() not implemeneted");
    }
    virtual bool tick();
    EpochResult run();
    void addPostBatchAction( PostBatchAction *action );
};


class DeepCL_EXPORT LearnBatcher : public Batcher {
public:
    float learningRate;
    LearnBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, float learningRate );
    virtual void _tick( float const*batchData, int const*batchLabels);
};


class DeepCL_EXPORT NetActionBatcher : public Batcher {
public:
    NetAction * netAction;
    NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction * netAction);
    virtual void _tick( float const*batchData, int const*batchLabels );
};


class DeepCL_EXPORT PropagateBatcher : public Batcher {
public:
    PropagateBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels);
    virtual void _tick( float const*batchData, int const*batchLabels);
};


class DeepCL_EXPORT NetLearnLabeledBatch : public NetAction {
public:
    float learningRate;
    NetLearnLabeledBatch( float learningRate ) :
        learningRate( learningRate ) {
    }   
    virtual void run( Trainable *net, float const*const batchData, int const*const batchLabels );
};


class DeepCL_EXPORT NetPropagateBatch : public NetAction {
public:
    NetPropagateBatch() {
    }
    virtual void run( Trainable *net, float const*const batchData, int const*const batchLabels );
};


class DeepCL_EXPORT NetBackpropBatch : public NetAction {
public:
    float learningRate;
    NetBackpropBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( Trainable *net, float const*const batchData, int const*const batchLabels );
};

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.

class DeepCL_EXPORT BatchLearner {
public:
    Trainable *net; // NOT owned by us, dont delete

    std::vector<PostBatchAction *> postBatchActions; // note: we DONT own these, dont delete, caller owns

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BatchLearner( Trainable *net );
    VIRTUAL void addPostBatchAction( PostBatchAction *action );
    EpochResult batchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction );
    EpochResult runBatchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction );
    int test( int batchSize, int N, float *testData, int const*testLabels );
    int propagateForTrain( int batchSize, int N, float *data, int const*labels );
    EpochResult backprop( float learningRate, int batchSize, int N, float *data, int const*labels );
    EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, float *trainData, int const*trainLabels );
    float runEpochFromExpected( float learningRate, int batchSize, int N, float *data, float *expectedResults );

    // [[[end]]]
};

