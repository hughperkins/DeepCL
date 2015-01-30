// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"

#include "BatchLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

void NetLearnLabeledBatch::run( NeuralNet *net, float *batchData, int *batchLabels ) {
    net->learnBatchFromLabels( learningRate, batchData, batchLabels );
}

void NetPropagateBatch::run( NeuralNet *net, float *batchData, int *batchLabels ) {
    net->propagate( batchData );
}

template< typename T > BatchLearner<T>::BatchLearner( NeuralNet *net, float dataTranslate, float dataScale ) :
    net( net ),
    dataTranslate( dataTranslate ),
    dataScale( dataScale ) {
}

template< typename T > EpochResult BatchLearner<T>::batchedNetAction( int batchSize, int N, T *data, int *labels, NetAction *netAction ) {
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

template< typename T > int BatchLearner<T>::test( int batchSize, int N, T *testData, int *testLabels ) {
    NetAction *action = new NetPropagateBatch();
    int numRight = batchedNetAction( batchSize, N, testData, testLabels, action ).numRight;
    delete action;
    return numRight;
}

template< typename T > EpochResult BatchLearner<T>::runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels ) {
    NetAction *action = new NetLearnLabeledBatch( learningRate );
    EpochResult epochResult = batchedNetAction( batchSize, Ntrain, trainData, trainLabels, action );
    delete action;
    return epochResult;
}

template class BatchLearner<unsigned char>;
template class BatchLearner<float>;

