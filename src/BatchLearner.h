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

class BatchLearner {
public:
    NeuralNet *net; // NOT owned by us, dont delete
    float dataTranslate;
    float dataScale;

    //float loss;
    //int numRight;

    int test( int batchSize, int N, unsigned char *testData, int *testLabels ) {
        int numRight = 0;
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
            unsigned char *thisBatchData = testData + batchStart * inputCubeSize;
            for( int i = 0; i < batchInputSize; i++ ) {
                batchData[i] = thisBatchData[i];
            }
            NormalizationHelper::normalize( batchData, batchInputSize, - dataTranslate, 1.0f / dataScale );
            net->propagate( batchData );
            numRight += net->calcNumRight( &(testLabels[batchStart]) );
        }
        delete[] batchData;
        return numRight;
    }

    template< typename T > EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels ) {
        const int inputCubeSize = net->getInputCubeSize();
        const int numBatches = ( Ntrain + batchSize - 1 ) / batchSize;
        float *batchData = new float[ batchSize * inputCubeSize ];
        float loss = 0;
        int numRight = 0;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int thisBatchSize = batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = Ntrain - (numBatches - 1) * batchSize;
                net->setBatchSize( thisBatchSize );
            }
            int batchStart = batchSize * batch;
            const int batchInputSize = thisBatchSize * inputCubeSize;
            T *thisBatchData = trainData + batchStart * inputCubeSize;
            for( int i = 0; i < batchInputSize; i++ ) {
                batchData[i] = thisBatchData[i];
            }
            NormalizationHelper::normalize( batchData, batchInputSize, - dataTranslate, 1.0f / dataScale );
            net->learnBatchFromLabels( learningRate, batchData, &(trainLabels[batchStart]) );
            loss += net->calcLossFromLabels( &(trainLabels[batchStart]) );
            numRight += net->calcNumRight( &(trainLabels[batchStart]) );
        }
        delete[] batchData;
        EpochResult epochResult( loss, numRight );
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

