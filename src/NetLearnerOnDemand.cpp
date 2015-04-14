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

#include "NetLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

NetLearnerOnDemand::NetLearnerOnDemand( Trainable *net ) :
        net( net ),
        batchLearnerOnDemand( net ) {
    batchSize = 128;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    learningDone = false;
    dumpTimings = false;
}

VIRTUAL NetLearnerOnDemand::~NetLearnerOnDemand() {
}

VIRTUAL void NetLearnerOnDemand::setTrainingData( std::string trainFilepath, int Ntrain ) {
    this->Ntrain = Ntrain;
    this->trainFilepath = trainFilepath;
}

VIRTUAL void NetLearnerOnDemand::setTestingData( std::string testFilepath, int Ntest ) {
    this->testFilepath = testFilepath;
    this->Ntest = Ntest;
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

VIRTUAL void NetLearnerOnDemand::setBatchSize( int fileReadBatches, int batchSize ) {
    this->batchSize = batchSize;
    this->fileReadBatches = fileReadBatches;
}

VIRTUAL void NetLearnerOnDemand::addPostEpochAction( PostEpochAction *action ) {
    postEpochActions.push_back( action );
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
}

VIRTUAL bool NetLearnerOnDemand::tickEpoch() {
    int epoch = nextEpoch;
    float annealedLearningRate = learningRate * pow( annealLearningRate, epoch );
    EpochResult epochResult = batchLearnerOnDemand.runEpochFromLabels( annealedLearningRate, trainFilepath, fileReadBatches, batchSize, Ntrain );
    cout << "dumpTimings " << dumpTimings << endl;
    if( dumpTimings ) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(epoch ) );
    cout << "annealed learning rate: " << annealedLearningRate << " training loss: " << epochResult.loss << endl;
    cout << " train accuracy: " << epochResult.numRight << "/" << Ntrain << " " << (epochResult.numRight * 100.0f/ Ntrain) << "%" << std::endl;
    int testNumRight = batchLearnerOnDemand.test( testFilepath, fileReadBatches, batchSize, Ntest );
    cout << "test accuracy: " << testNumRight << "/" << Ntest << " " << (testNumRight * 100.0f / Ntest ) << "%" << endl;
    timer.timeCheck("after tests");
    for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
        (*it)->run( epoch );
    }
    nextEpoch++;
    if( nextEpoch == numEpochs ) {
        learningDone = true;
    }
    return !learningDone;
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


