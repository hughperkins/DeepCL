// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearnerOnDemand.h"
#include "NeuralNet.h"
#include "Trainable.h"
#include "NetAction.h"
#include "OnDemandBatcher.h"

#include "NetLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

NetLearnerOnDemand::NetLearnerOnDemand( Trainable *net ) :
        net( net )
//    batchSize = 128;
        {
    learnAction = new NetLearnLabeledAction( 0 );
    testAction = new NetPropagateAction();
    learnBatcher = new OnDemandBatcher( net, learnAction, "", 0, 0, 0 );
    testBatcher = new OnDemandBatcher( net, testAction, "", 0, 0, 0 );
    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    learningDone = false;
    dumpTimings = false;
}

VIRTUAL NetLearnerOnDemand::~NetLearnerOnDemand() {
    delete learnBatcher;
    delete testBatcher;
    delete testAction;
    delete learnAction;
}

VIRTUAL void NetLearnerOnDemand::setTrainingData( std::string trainFilepath, int Ntrain ) {
    //this->Ntrain = Ntrain;
    //this->trainFilepath = trainFilepath;
    learnBatcher->filepath = trainFilepath;
    learnBatcher->N = Ntrain;
}

VIRTUAL void NetLearnerOnDemand::setTestingData( std::string testFilepath, int Ntest ) {
//    this->testFilepath = testFilepath;
//    this->Ntest = Ntest;
    testBatcher->filepath = testFilepath;
    testBatcher->N = Ntest;
}

VIRTUAL void NetLearnerOnDemand::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 1 );
}

VIRTUAL void NetLearnerOnDemand::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

VIRTUAL void NetLearnerOnDemand::setSchedule( int numEpochs, int nextEpoch ) {
    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}

VIRTUAL bool NetLearnerOnDemand::isEpochDone() {
    return learnBatcher->epochDone;
}

VIRTUAL int NetLearnerOnDemand::getNextEpoch() {
    return nextEpoch;
}

VIRTUAL void NetLearnerOnDemand::setBatchSize( int fileReadBatches, int batchSize ) {
//    this->batchSize = batchSize;
//    this->fileReadBatches = fileReadBatches;
    learnBatcher->batchSize = batchSize;
    testBatcher->batchSize = batchSize;
    learnBatcher->fileReadBatches = fileReadBatches;
    testBatcher->fileReadBatches = fileReadBatches;
}

VIRTUAL void NetLearnerOnDemand::setLearningRate( float learningRate ) {
    this->setLearningRate( learningRate, 1.0f );
}

VIRTUAL void NetLearnerOnDemand::setLearningRate( float learningRate, float annealLearningRate ) {
    this->learningRate = learningRate;
    this->annealLearningRate = annealLearningRate;
}

VIRTUAL void NetLearnerOnDemand::reset() {
    timer.lap();
    learningDone = false;
    nextEpoch = 0;
    learnBatcher->reset();
    testBatcher->reset();
}

VIRTUAL void NetLearnerOnDemand::postEpochTesting() {
    cout << "dumpTimings " << dumpTimings << endl;
    if( dumpTimings ) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch + 1 ) );
    cout << "annealed learning rate: " << learnAction->learningRate << " training loss: " << learnBatcher->loss << endl;
    cout << " train accuracy: " << learnBatcher->numRight << "/" << learnBatcher->N << " " << (learnBatcher->numRight * 100.0f/ learnBatcher->N) << "%" << std::endl;
    testBatcher->run();
//    int testNumRight = batchLearnerOnDemand.test( testFilepath, fileReadBatches, batchSize, Ntest );
    cout << "test accuracy: " << testBatcher->numRight << "/" << testBatcher->N << " " << (testBatcher->numRight * 100.0f / testBatcher->N ) << "%" << endl;
    timer.timeCheck("after tests");
}

VIRTUAL bool NetLearnerOnDemand::tickBatch() { // means: filebatch, not low-level batch
                                               // probalby good enough for now?    
    int epoch = nextEpoch;
    learnAction->learningRate = learningRate * pow( annealLearningRate, epoch );
    learnBatcher->tick();       // returns false once all learning done (all epochs)
    if( learnBatcher->epochDone ) {
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

VIRTUAL bool NetLearnerOnDemand::tickEpoch() {
    int epoch = nextEpoch;
    cout << "NetLearnerOnDemand.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << learnBatcher->epochDone << endl;
    cout << "numEpochs=" << numEpochs << endl;
    if( learnBatcher->epochDone ) {
        learnBatcher->reset();
    }
    while(!learnBatcher->epochDone ) {
        tickBatch();
    }
    return !learningDone;

//    int epoch = nextEpoch;
//    learnAction->learningRate = learningRate * pow( annealLearningRate, epoch );
//    learnBatcher->tick();       // returns false once all learning done (all epochs)
//    if( learnBatcher->epochDone ) {
//        postEpochTesting();
//        nextEpoch++;
//    }
//    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
//    if( nextEpoch == numEpochs ) {
//        cout << "setting learningdone to true" << endl;
//        learningDone = true;
//    }
//    return !learningDone;
}

VIRTUAL void NetLearnerOnDemand::run() {
    if( learningDone ) {
        reset();
    }
    while( !learningDone ) {
        tickEpoch();
    }
}

VIRTUAL bool NetLearnerOnDemand::isLearningDone() {
    return learningDone;
}

VIRTUAL void NetLearnerOnDemand::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}

VIRTUAL void NetLearnerOnDemand::learn( float learningRate, float annealLearningRate ) {
    setLearningRate( learningRate, annealLearningRate );
    run();
}


