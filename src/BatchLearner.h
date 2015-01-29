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
    virtual void run( NeuralNet *net, float *batchData, int *batchLabels ) {
        net->learnBatchFromLabels( learningRate, batchData, batchLabels );
    }
};

class NetPropagateBatch : public NetAction {
public:
    NetPropagateBatch() {
    }
    virtual void run( NeuralNet *net, float *batchData, int *batchLabels ) {
        net->propagate( batchData );
    }
};

class BatchLearner {
public:
    NeuralNet *net; // NOT owned by us, dont delete
    float dataTranslate;
    float dataScale;

    template<typename T>
    EpochResult batchedNetAction( int batchSize, int N, T *data, int *labels, NetAction *netAction ) {
        int numRight = 0;
        float loss = 0;
        net->setBatchSize( batchSize );
        int numBatches = (N + batchSize - 1 ) / batchSize;
        int inputCubeSize = net->getInputCubeSize();
        float *batchData = new float[ batchSize * inputCubeSize ];
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * batchSize;
            int thisBatchSize = batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = N - batchStart;
                net->setBatchSize( thisBatchSize );
            }
            const int batchInputSize = thisBatchSize * inputCubeSize;
            T *thisBatchData = data + batchStart * inputCubeSize;
            for( int i = 0; i < batchInputSize; i++ ) {
                batchData[i] = thisBatchData[i];
            }
            NormalizationHelper::normalize( batchData, batchInputSize, - dataTranslate, 1.0f / dataScale );
            netAction->run( net, batchData, &(labels[batchStart]) );
            loss += net->calcLossFromLabels( &(labels[batchStart]) );
            numRight += net->calcNumRight( &(labels[batchStart]) );
        }
        delete[] batchData;
        EpochResult epochResult( loss, numRight );
        return epochResult;
    }

    int test( int batchSize, int N, unsigned char *testData, int *testLabels ) {
        NetAction *action = new NetPropagateBatch();
        int numRight = batchedNetAction( batchSize, N, testData, testLabels, action ).numRight;
        delete action;
        return numRight;
    }

    template< typename T > EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels ) {
        NetAction *action = new NetLearnLabeledBatch( learningRate );
        EpochResult epochResult = batchedNetAction( batchSize, Ntrain, trainData, trainLabels, action );
        delete action;
        return epochResult;
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BatchLearner( NeuralNet *net, float dataTranslate, float dataScale );

    // [[[end]]]
};

