// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

class NeuralNet;
class Trainable;

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

class DeepCL_EXPORT PostBatchAction {
public:
    virtual void run( int batch, float lossSoFar, int numRightSoFar ) = 0;
};

class DeepCL_EXPORT EpochResult {
public:
    float loss;
    int numRight;
    EpochResult( float loss, int numRight ) :
        loss( loss ),
        numRight( numRight ) {
    }
};

template< typename T>
class DeepCL_EXPORT NetAction {
public:
    virtual ~NetAction() {}
    virtual void run( Trainable *net, T *batchData, int const*batchLabels ) = 0;
};

template< typename T>
class DeepCL_EXPORT NetLearnLabeledBatch : public NetAction<T> {
public:
    float learningRate;
    NetLearnLabeledBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( Trainable *net, T *batchData, int const*batchLabels );
};

template< typename T>
class DeepCL_EXPORT NetPropagateBatch : public NetAction<T> {
public:
    NetPropagateBatch() {
    }
    virtual void run( Trainable *net, T *batchData, int const*batchLabels );
};

template< typename T>
class DeepCL_EXPORT NetBackpropBatch : public NetAction<T> {
public:
    float learningRate;
    NetBackpropBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( Trainable *net, T *batchData, int const*batchLabels );
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
    int test( int batchSize, int N, T *testData, int const*testLabels );
    int propagateForTrain( int batchSize, int N, T *data, int const*labels );
    EpochResult backprop( float learningRate, int batchSize, int N, T *data, int const*labels );
    EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int const*trainLabels );
    float runEpochFromExpected( float learningRate, int batchSize, int N, T *data, float *expectedResults );

    // [[[end]]]
};

