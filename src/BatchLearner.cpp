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

template< typename T>
void NetLearnLabeledBatch<T>::run( NeuralNet *net, T *batchData, int *batchLabels ) {
    net->learnBatchFromLabels( learningRate, batchData, batchLabels );
}

template< typename T>
void NetPropagateBatch<T>::run( NeuralNet *net, T *batchData, int *batchLabels ) {
    net->propagate( batchData );
}

template< typename T > BatchLearner<T>::BatchLearner( NeuralNet *net ) :
    net( net ) {
}

template< typename T > EpochResult BatchLearner<T>::batchedNetAction( int batchSize, int N, T *data, int *labels, NetAction<T> *netAction ) {
    int numRight = 0;
    float loss = 0;
    net->setBatchSize( batchSize );
    int numBatches = (N + batchSize - 1 ) / batchSize;
    int inputCubeSize = net->getInputCubeSize();
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        if( batch == numBatches - 1 ) {
            net->setBatchSize( N - batchStart );
        }
        netAction->run( net, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
        loss += net->calcLossFromLabels( &(labels[batchStart]) );
        numRight += net->calcNumRight( &(labels[batchStart]) );
    }
    EpochResult epochResult( loss, numRight );
    return epochResult;
}

template< typename T > int BatchLearner<T>::test( int batchSize, int N, T *testData, int *testLabels ) {
    net->setTraining( false );
    NetAction<T> *action = new NetPropagateBatch<T>();
    int numRight = batchedNetAction( batchSize, N, testData, testLabels, action ).numRight;
    delete action;
    return numRight;
}

template< typename T > EpochResult BatchLearner<T>::runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int *trainLabels ) {
    net->setTraining( true );
    NetAction<T> *action = new NetLearnLabeledBatch<T>( learningRate );
    EpochResult epochResult = batchedNetAction( batchSize, Ntrain, trainData, trainLabels, action );
    delete action;
    return epochResult;
}

template class BatchLearner<unsigned char>;
template class BatchLearner<float>;

