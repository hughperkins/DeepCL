// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearner.h"
#include "NeuralNet.h"
#include "Trainable.h"
#include "NetAction.h"

#include "NetLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

NetLearner::NetLearner( Trainable *net ) :
        net( net )
        {
//    batchSize = 128;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    dumpTimings = false;
//    batcher = 0;
//    learnAction = 0;
    learningDone = false;

    trainBatcher = new LearnBatcher( net, 0, 0, 0, 0, 0 );
    testBatcher = new PropagateBatcher( net, 0, 0, 0, 0 );   
//    reset();
}

VIRTUAL NetLearner::~NetLearner() {
    delete trainBatcher;
    delete testBatcher;
}

VIRTUAL void NetLearner::setTrainingData( int Ntrain, float *trainData, int *trainLabels ) {
    this->trainBatcher->N = Ntrain;
    this->trainBatcher->data = trainData;
    this->trainBatcher->labels = trainLabels;
//    cout << "NetLearner.settrainingdata data=" << (void *)trainData << endl;
}

VIRTUAL void NetLearner::setTestingData( int Ntest, float *testData, int *testLabels ) {
    this->testBatcher->N = Ntest;
    this->testBatcher->data = testData;
    this->testBatcher->labels = testLabels;
}

VIRTUAL void NetLearner::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 0 );
}

VIRTUAL void NetLearner::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

VIRTUAL void NetLearner::setSchedule( int numEpochs, int nextEpoch ) {
    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}

VIRTUAL void NetLearner::setBatchSize( int batchSize ) {
    this->trainBatcher->batchSize = batchSize;
    this->testBatcher->batchSize = batchSize;
}

VIRTUAL void NetLearner::reset() {
    cout << "NetLearner::reset()" << endl;
    learningDone = false;
    nextEpoch = 0;
//    net->setTraining( true );
    trainBatcher->reset();
    testBatcher->reset();
    timer.lap();
}

VIRTUAL void NetLearner::postEpochTesting() {
    if( dumpTimings ) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch+1) );
    cout << "annealed learning rate: " << trainBatcher->learningRate << " training loss: " << trainBatcher->loss << endl;
    cout << " train accuracy: " << trainBatcher->numRight << "/" << trainBatcher->N << " " << (trainBatcher->numRight * 100.0f/ trainBatcher->N) << "%" << std::endl;
    testBatcher->run();
    cout << "test accuracy: " << testBatcher->numRight << "/" << testBatcher->N << " " << 
        (testBatcher->numRight * 100.0f / testBatcher->N ) << "%" << endl;
    timer.timeCheck("after tests");
}

VIRTUAL bool NetLearner::tickBatch() { // just tick one learn batch, once all done, then run testing etc
    int epoch = nextEpoch;
    trainBatcher->learningRate = learningRate * pow( annealLearningRate, epoch );
    trainBatcher->tick();       // returns false once all learning done (all epochs)
    if( trainBatcher->epochDone ) {
        postEpochTesting();
        nextEpoch++;
    }
    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
    if( nextEpoch == numEpochs ) {
        cout << "setting learningdone to true" << endl;
        learningDone = true;
    }
    return !learningDone;
}

VIRTUAL bool NetLearner::isEpochDone() {
    return trainBatcher->epochDone;
}

VIRTUAL int NetLearner::getNextEpoch() {
    return nextEpoch;
}

VIRTUAL int NetLearner::getNextBatch() {
    return trainBatcher->getNextBatch();
}

VIRTUAL int NetLearner::getBatchNumRight() {
    return trainBatcher->numRight;
}

VIRTUAL float NetLearner::getBatchLoss() {
    return trainBatcher->loss;
}

VIRTUAL void NetLearner::setBatchState( int batch, int numRight, float loss ) {
    trainBatcher->nextBatch = batch;
    trainBatcher->numRight = numRight;
    trainBatcher->loss = loss;
}

VIRTUAL bool NetLearner::tickEpoch() {
    int epoch = nextEpoch;
    cout << "NetLearner.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << trainBatcher->epochDone << endl;
    cout << "numEpochs=" << numEpochs << endl;
    if( trainBatcher->epochDone ) {
        trainBatcher->reset();
    }
    while(!trainBatcher->epochDone ) {
        tickBatch();
    }
    return !learningDone;
}

VIRTUAL void NetLearner::run() {
    if( learningDone ) {
        reset();
    }
    while( !learningDone ) {
        tickEpoch();
    }
}

VIRTUAL bool NetLearner::isLearningDone() {
    return learningDone;
}

VIRTUAL void NetLearner::setLearningRate( float learningRate ) {
    this->setLearningRate( learningRate, 1.0f );
}

VIRTUAL void NetLearner::setLearningRate( float learningRate, float annealLearningRate ) {
    this->learningRate = learningRate;
    this->annealLearningRate = annealLearningRate;
}

VIRTUAL void NetLearner::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}
VIRTUAL void NetLearner::learn( float learningRate, float annealLearningRate ) {
    setLearningRate( learningRate, annealLearningRate );
    run();
}


