// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "Trainable.h"

#include "BatchLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Batcher::Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels ) :
        net(net),
        batchSize(batchSize),
        N(N),
        data(data),
        labels(labels)
            {
    inputCubeSize = net->getInputCubeSize();
    updateVars();
    reset();
}

void Batcher::updateVars() {
    if( batchSize != 0 ) {
        numBatches = ( N + batchSize - 1 ) / batchSize;
    }
}

void Batcher::reset() {
    nextBatch = 0;
    numRight = 0;
    loss = 0;
    epochDone = false;
}
//
//void Batcher::_tick( float const*batchData, int const*batchLabels ) {
//}

bool Batcher::tick() {
    updateVars();
    int batch = nextBatch;
//    std::cout << "BatchLearner.tick() batch=" << batch << std::endl;
    int batchStart = batch * batchSize;
    int thisBatchSize = batchSize;
    if( batch == numBatches - 1 ) {
        thisBatchSize = N - batchStart;
    }
//    std::cout << "batchSize=" << batchSize << " thisBatchSize=" << thisBatchSize << " batch=" << batch <<
//            " batchStart=" << batchStart << " data=" << (void *)data << " labels=" << labels << 
//            std::endl;
    net->setBatchSize( thisBatchSize );
    internalTick(  &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
//        netAction->run( net, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]) );
    float thisLoss = net->calcLossFromLabels( &(labels[batchStart]) );
    int thisNumRight = net->calcNumRight( &(labels[batchStart]) );
//        std::cout << "thisloss " << thisLoss << " thisnumright " << thisNumRight << std::endl; 
    loss += thisLoss;
    numRight += thisNumRight;
    nextBatch++;
    if( nextBatch == numBatches ) {
        epochDone = true;
    }
    return !epochDone;
}

EpochResult Batcher::run() {
    if( data == 0 ) {
        throw runtime_error("Batcher: no data set");
    }
    if( labels == 0 ) {
        throw runtime_error("Batcher: no labels set");
    }
    if( epochDone ) {
        reset();
    }
    while( !epochDone ) {
        tick();
    }
    EpochResult epochResult( loss, numRight );
    return epochResult;
}

void NetLearnLabeledBatch::run( Trainable *net, float const*const batchData, int const*const batchLabels ) {
//    cout << "NetLearnLabeledBatch learningrate=" << learningRate << endl;
    net->learnBatchFromLabels( learningRate, batchData, batchLabels );
}


LearnBatcher::LearnBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, float learningRate ) :
    Batcher( net, batchSize, N, data, labels ),
    learningRate( learningRate ) {
}


void LearnBatcher::internalTick( float const*batchData, int const*batchLabels) {
//    cout << "LearnBatcher learningRate=" << learningRate << " batchdata=" << (void *)batchData << 
//        " batchLabels=" << batchLabels << endl;
    net->learnBatchFromLabels( learningRate, batchData, batchLabels );
}
 

NetActionBatcher::NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction *netAction) :
    Batcher( net, batchSize, N, data, labels ),
    netAction( netAction ) {
}


void NetActionBatcher::internalTick( float const*batchData, int const*batchLabels ) {
    netAction->run( this->net, batchData, batchLabels );
}


PropagateBatcher::PropagateBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels ) :
    Batcher( net, batchSize, N, data, labels ) {
}


void PropagateBatcher::internalTick( float const*batchData, int const*batchLabels) {
    this->net->propagate( batchData );
}


void NetPropagateBatch::run( Trainable *net, float const*const batchData, int const*const batchLabels ) {
//    cout << "NetPropagateBatch" << endl;
    net->propagate( batchData );
}


void NetBackpropBatch::run( Trainable *net, float const*const batchData, int const*const batchLabels ) {
//    cout << "NetBackpropBatch learningrate=" << learningRate << endl;
    net->backPropFromLabels( learningRate, batchLabels );
}

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
    NetPropagateBatch *action = new NetPropagateBatch();
    int numRight = runBatchedNetAction( batchSize, N, testData, testLabels, action ).numRight;
    delete action;
    return numRight;
}

 int BatchLearner::propagateForTrain( int batchSize, int N, float *data, int const*labels ) {
    net->setTraining( true );
    NetPropagateBatch *action = new NetPropagateBatch();
    int numRight = runBatchedNetAction( batchSize, N, data, labels, action ).numRight;
    delete action;
    return numRight;
}

 EpochResult BatchLearner::backprop( float learningRate, int batchSize, int N, float *data, int const*labels ) {
    net->setTraining( true );
    NetBackpropBatch *action = new NetBackpropBatch( learningRate );
    EpochResult epochResult = runBatchedNetAction( batchSize, N, data, labels, action );
    delete action;
    return epochResult;
}

 EpochResult BatchLearner::runEpochFromLabels( float learningRate, int batchSize, int Ntrain, float *trainData, int const*trainLabels ) {
    net->setTraining( true );
    NetLearnLabeledBatch *action = new NetLearnLabeledBatch( learningRate );
    EpochResult epochResult = runBatchedNetAction( batchSize, Ntrain, trainData, trainLabels, action );
    delete action;
    return epochResult;
}

 float BatchLearner::runEpochFromExpected( float learningRate, int batchSize, int N, float *data, float *expectedResults ) {
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
        net->learnBatch( learningRate, &(data[ batchStart * inputCubeSize ]), &(expectedResults[batchStart * outputCubeSize]) );
        loss += net->calcLoss( &( expectedResults[batchStart * outputCubeSize]) );
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
//        net->learnBatch( learningRate, &(data[ batchStart * inputCubeSize ]), &(expectedResults[batchStart * outputCubeSize]) );
//        loss += net->calcLoss( &( expectedResults[batchStart * outputCubeSize]) );
//        numRight += AccuracyHelper::calcNumRight( thisBatchSize, net->getLayerLayer()->getOutputPlanes(), &( labels[ batchStart] ), net->getResults() );
//    }
//    EpochResult epochResult( loss, numRight );
//    return epochResult;
//}


