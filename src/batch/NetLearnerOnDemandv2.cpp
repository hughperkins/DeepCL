// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "util/StatefulTimer.h"
#include "util/Timer.h"
#include "batch/BatchLearnerOnDemand.h"
#include "net/NeuralNet.h"
#include "net/Trainable.h"
#include "batch/NetAction.h"
#include "batch/OnDemandBatcherv2.h"
#include "util/stringhelper.h"
//#include "loaders/GenericLoaderv2.h"
#include "batch/NetLearnerOnDemandv2.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI NetLearnerOnDemandv2::NetLearnerOnDemandv2(Trainer *trainer, Trainable *net, 
            GenericLoaderv2 *trainLoader, int Ntrain,
            GenericLoaderv2 *validateLoader, int Ntest,
            int fileReadBatches, int batchSize) :
        net(net),
        learnBatcher(0),
        testBatcher(0)
//    batchSize = 128;
        {
    learnAction = new NetLearnLabeledAction(trainer);
    testAction = new NetForwardAction();
    learnBatcher = new OnDemandBatcherv2(net, learnAction, trainLoader, Ntrain, fileReadBatches, batchSize);
    testBatcher = new OnDemandBatcherv2(net, testAction, validateLoader, Ntest, fileReadBatches, batchSize);
//    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    learningDone = false;
    dumpTimings = false;
}
VIRTUAL NetLearnerOnDemandv2::~NetLearnerOnDemandv2() {
    if(learnBatcher != 0) {
        delete learnBatcher;
    }
    if(testBatcher != 0) {
        delete testBatcher;
    }
    delete testAction;
    delete learnAction;
}
VIRTUAL void NetLearnerOnDemandv2::setSchedule(int numEpochs) {
    setSchedule(numEpochs, 1);
}
VIRTUAL void NetLearnerOnDemandv2::setDumpTimings(bool dumpTimings) {
    this->dumpTimings = dumpTimings;
}
VIRTUAL void NetLearnerOnDemandv2::setSchedule(int numEpochs, int nextEpoch) {
    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::getEpochDone() {
    return learnBatcher->getEpochDone();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNextEpoch() {
    return nextEpoch;
}
//VIRTUAL void NetLearnerOnDemandv2::setLearningRate(float learningRate) {
//    this->setLearningRate(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemandv2::setLearningRate(float learningRate, float annealLearningRate) {
//    this->learningRate = learningRate;
//    this->annealLearningRate = annealLearningRate;
//}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNextBatch() {
    return learnBatcher->getNextBatch();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNTrain() {
    return learnBatcher->getN();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getBatchNumRight() {
    return learnBatcher->getNumRight();
}
PUBLICAPI VIRTUAL float NetLearnerOnDemandv2::getBatchLoss() {
    return learnBatcher->getLoss();
}
VIRTUAL void NetLearnerOnDemandv2::setBatchState(int nextBatch, int numRight, float loss) {
    learnBatcher->setBatchState(nextBatch, numRight, loss);
}
PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::reset() {
    timer.lap();
    learningDone = false;
    nextEpoch = 0;
    learnBatcher->reset();
    testBatcher->reset();
}
VIRTUAL void NetLearnerOnDemandv2::postEpochTesting() {
    cout << "dumpTimings " << dumpTimings << endl;
    if(dumpTimings) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch + 1) );
//    cout << "annealed learning rate: " << learnAction->getLearningRate()
    cout << " training loss: " << learnBatcher->getLoss() << endl;
    cout << " train accuracy: " << learnBatcher->getNumRight() << "/" << learnBatcher->getN() << " " << (learnBatcher->getNumRight() * 100.0f/ learnBatcher->getN()) << "%" << std::endl;
    testBatcher->run(nextEpoch);
//    int testNumRight = batchLearnerOnDemand.test(testFilepath, fileReadBatches, batchSize, Ntest);
    cout << "test accuracy: " << testBatcher->getNumRight() << "/" << testBatcher->getN() << " " << (testBatcher->getNumRight() * 100.0f / testBatcher->getN()) << "%" << endl;
    timer.timeCheck("after tests");
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::tickBatch() { // means: filebatch, not low-level batch
                                               // probalby good enough for now?    
//    int epoch = nextEpoch;
//    learnAction->learningRate = learningRate * pow(annealLearningRate, epoch);
    learnBatcher->tick(nextEpoch);       // returns false once all learning done (all epochs)
    if(learnBatcher->getEpochDone()) {
        postEpochTesting();
        nextEpoch++;
    }
//    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
    if(nextEpoch == numEpochs) {
//        cout << "setting learningdone to true" << endl;
        learningDone = true;
    }
    return !learningDone;
}

PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::tickEpoch() {
//    int epoch = nextEpoch;
//    cout << "NetLearnerOnDemandv2.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << learnBatcher->getEpochDone() << endl;
//    cout << "numEpochs=" << numEpochs << endl;
    if(learnBatcher->getEpochDone()) {
        learnBatcher->reset();
    }
    while(!learnBatcher->getEpochDone()) {
        tickBatch();
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::run() {
    if(learningDone) {
        reset();
    }
    while(!learningDone) {
        tickEpoch();
    }
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::isLearningDone() {
    return learningDone;
}
//PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::learn(float learningRate) {
//    learn(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemandv2::learn(float learningRate, float annealLearningRate) {
//    setLearningRate(learningRate, annealLearningRate);
//    run();
//}
//VIRTUAL void NetLearnerOnDemandv2::setTrainer(Trainer *trainer) {
//    this->trainer = trainer;
//}

