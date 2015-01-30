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

#define VIRTUAL virtual
#define STATIC static

class EpochResult {
public:
    float loss;
    int numRight;
    EpochResult( float loss, int numRight ) :
        loss( loss ),
        numRight( numRight ) {
    }
};

class NetAction {
public:
    virtual void run( NeuralNet *net, float *batchData, int *batchLabels ) = 0;
};

class NetLearnLabeledBatch : public NetAction {
public:
    float learningRate;
    NetLearnLabeledBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( NeuralNet *net, float *batchData, int *batchLabels );
};

class NetPropagateBatch : public NetAction {
public:
    NetPropagateBatch() {
    }
    virtual void run( NeuralNet *net, float *batchData, int *batchLabels );
};

template< typename T>
class BatchLearner {
public:
    NeuralNet *net; // NOT owned by us, dont delete
    float dataTranslate;
    float dataScale;


    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearner( NeuralNet *net, float dataTranslate, float dataScale );
    EpochResult batchedNetAction( int batchSize, int N, T *data, int *labels, NetAction *netAction );
    int test( int batchSize, int N, T *testData, int *testLabels );
    EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels );

    // [[[end]]]
};

