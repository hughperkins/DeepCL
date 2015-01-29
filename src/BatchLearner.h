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

class BatchLearner {
public:
    NeuralNet *net; // NOT owned by us, dont delete
    float dataTranslate;
    float dataScale;

    float loss;
    int numRight;


    template< typename T > void runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels ) {
        const int inputCubeSize = net->getInputCubeSize();
        const int numBatches = ( Ntrain + batchSize - 1 ) / batchSize;
        float *batchData = new float[ batchSize * inputCubeSize ];
        loss = 0;
        numRight = 0;
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
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BatchLearner( NeuralNet *net, float dataTranslate, float dataScale );
    VIRTUAL float getLoss() const;
    VIRTUAL int getNumRight() const;

    // [[[end]]]
};

