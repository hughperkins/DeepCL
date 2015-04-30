// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

//#include "NormalizationHelper.h"
//#include "NeuralNet.h"
//#include "AccuracyHelper.h"
#include "Trainable.h"
#include "NetAction.h"
#include "BatchLearner.h"
#include "Batcher.h"
#include "Trainer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

BatchLearner::BatchLearner( Trainable *net ) :
    net( net ) {
}

EpochResult BatchLearner::batchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction ) {
    return runBatchedNetAction( batchSize, N, data, labels, netAction );
}

EpochResult BatchLearner::runBatchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction ) {
    NetActionBatcher batcher(net, batchSize, N, data, labels, netAction);
    return batcher.run();
}

int BatchLearner::test( int batchSize, int N, float *testData, int const*testLabels ) {
    net->setTraining( false );
    NetForwardAction *action = new NetForwardAction();
    int numRight = runBatchedNetAction( batchSize, N, testData, testLabels, action ).numRight;
    delete action;
    return numRight;
}

int BatchLearner::forwardForTrain( int batchSize, int N, float *data, int const*labels ) {
    net->setTraining( true );
    NetForwardAction *action = new NetForwardAction();
    int numRight = runBatchedNetAction( batchSize, N, data, labels, action ).numRight;
    delete action;
    return numRight;
}

//EpochResult BatchLearner::backprop( float learningRate, int batchSize, int N, float *data, int const*labels ) {
//    net->setTraining( true );
//    NetBackpropAction *action = new NetBackpropAction( learningRate );
//    EpochResult epochResult = runBatchedNetAction( batchSize, N, data, labels, action );
//    delete action;
//    return epochResult;
//}

EpochResult BatchLearner::runEpochFromLabels( Trainer *trainer, int batchSize, int Ntrain, float *trainData, int const*trainLabels ) {
    net->setTraining( true );
    NetLearnLabeledAction *action = new NetLearnLabeledAction( trainer );
    EpochResult epochResult = runBatchedNetAction( batchSize, Ntrain, trainData, trainLabels, action );
    delete action;
    return epochResult;
}

float BatchLearner::runEpochFromExpected( Trainer *trainer, int batchSize, int N, float *data, float *expectedOutput ) {
    net->setTraining( true );
    float loss = 0;
    net->setBatchSize( batchSize );
    const int numBatches = (N + batchSize - 1 ) / batchSize;
    const int inputCubeSize = net->getInputCubeSize();
    const int outputCubeSize = net->getOutputCubeSize();
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        if( batch == numBatches - 1 ) {
            net->setBatchSize( N - batchStart );
        }
        trainer->train( net, &(data[ batchStart * inputCubeSize ]), &(expectedOutput[batchStart * outputCubeSize]) );
        loss += net->calcLoss( &( expectedOutput[batchStart * outputCubeSize]) );
    }
    return loss;
}

// EpochResult BatchLearner::runEpochFromExpectedWithLabels( float learningRate, int batchSize, int Ntrain, float *trainData, float *expectedValues, int *labels ) {
//    net->setTraining( true );
//    int numRight = 0;
//    float loss = 0;
//    net->setBatchSize( batchSize );
//    const int numBatches = (N + batchSize - 1 ) / batchSize;
//    const int inputCubeSize = net->getInputCubeSize();
//    const int outputCubeSize = net->getOutputCubeSize();
//    for( int batch = 0; batch < numBatches; batch++ ) {
//        int batchStart = batch * batchSize;
//        if( batch == numBatches - 1 ) {
//            net->setBatchSize( N - batchStart );
//        }
//        net->learnBatch( learningRate, &(data[ batchStart * inputCubeSize ]), &(expectedOutput[batchStart * outputCubeSize]) );
//        loss += net->calcLoss( &( expectedOutput[batchStart * outputCubeSize]) );
//        numRight += AccuracyHelper::calcNumRight( thisBatchSize, net->getLayerLayer()->getOutputPlanes(), &( labels[ batchStart] ), net->getOutput() );
//    }
//    EpochResult epochResult( loss, numRight );
//    return epochResult;
//}


