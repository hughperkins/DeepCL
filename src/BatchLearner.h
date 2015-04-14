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

template< typename T>
class DeepCL_EXPORT NetAction {
public:
    virtual ~NetAction() {}
    virtual void run( Trainable *net, T const*const batchData, int const*const batchLabels ) = 0;
};

template< typename T>
class DeepCL_EXPORT Batcher  {
public:
    Trainable *net;
    int batchSize;
    int N;
    T const* data;
    int const* labels;
    NetAction<T> * netAction;

    int numBatches;
    int inputCubeSize;

    bool epochDone;
    int nextBatch;
    int numRight;
    float loss;

    std::vector<PostBatchAction *> postBatchActions; // note: we DONT own these, dont delete, caller owns

    Batcher(Trainable *net, int batchSize, int N, T *data, int const*labels, NetAction<T> *netAction) :
            net(net),
            batchSize(batchSize),
            N(N),
            data(data),
            labels(labels),
            netAction(netAction),
            inputCubeSize( net->getInputCubeSize() )
        {
        updateVars();
        reset();
    }
    void updateVars() {
        if( batchSize != 0 ) {
            numBatches = ( N + batchSize - 1 ) / batchSize;
        }
    }
    void reset() {
        nextBatch = 0;
        numRight = 0;
        loss = 0;
        epochDone = false;
    }
    bool tick() {
        updateVars();
        int batch = nextBatch;
//        std::cout << "BatchLearner.tick() batch=" << batch << std::endl;
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = N - batchStart;
            net->setBatchSize( thisBatchSize );
        }
//        std::cout << "batchSize=" << batchSize << " thisBatchSize=" << thisBatchSize << " batch=" << batch <<
//            " batchStart=" << batchStart << " data=" << (void *)data << " labels=" << labels << std::endl;
        net->setBatchSize( thisBatchSize );
        netAction->run( net, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
        float thisLoss = net->calcLossFromLabels( &(labels[batchStart]) );
        int thisNumRight = net->calcNumRight( &(labels[batchStart]) );
//        std::cout << "thisloss " << thisLoss << " thisnumright " << thisNumRight << std::endl; 
        loss += thisLoss;
        numRight += thisNumRight;
        for( std::vector<PostBatchAction *>::iterator it = postBatchActions.begin(); it != postBatchActions.end(); it++ ) {
            (*it)->run( batch, numRight, loss );
        }
        nextBatch++;
        if( nextBatch == numBatches ) {
            epochDone = true;
        }
        return !epochDone;
    }
    EpochResult run() {
        if( epochDone ) {
            reset();
        }
        while( !epochDone ) {
            tick();
        }
        EpochResult epochResult( loss, numRight );
        return epochResult;
    }
    void addPostBatchAction( PostBatchAction *action ) {
        postBatchActions.push_back( action );
    }
};

template< typename T>
class DeepCL_EXPORT NetLearnLabeledBatch : public NetAction<T> {
public:
    float learningRate;
    NetLearnLabeledBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( Trainable *net, T const*const batchData, int const*const batchLabels );
};

template< typename T>
class DeepCL_EXPORT NetPropagateBatch : public NetAction<T> {
public:
    NetPropagateBatch() {
    }
    virtual void run( Trainable *net, T const*const batchData, int const*const batchLabels );
};

template< typename T>
class DeepCL_EXPORT NetBackpropBatch : public NetAction<T> {
public:
    float learningRate;
    NetBackpropBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( Trainable *net, T const*const batchData, int const*const batchLabels );
};

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.
template< typename T>
class DeepCL_EXPORT BatchLearner {
public:
    Trainable *net; // NOT owned by us, dont delete

    std::vector<PostBatchAction *> postBatchActions; // note: we DONT own these, dont delete, caller owns

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearner( Trainable *net );
    VIRTUAL void addPostBatchAction( PostBatchAction *action );
    EpochResult batchedNetAction( int batchSize, int N, T *data, int const*labels, NetAction<T> *netAction );
    EpochResult runBatchedNetAction( int batchSize, int N, T *data, int const*labels, NetAction<T> *netAction );
    int test( int batchSize, int N, T *testData, int const*testLabels );
    int propagateForTrain( int batchSize, int N, T *data, int const*labels );
    EpochResult backprop( float learningRate, int batchSize, int N, T *data, int const*labels );
    EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int const*trainLabels );
    float runEpochFromExpected( float learningRate, int batchSize, int N, T *data, float *expectedResults );

    // [[[end]]]
};

