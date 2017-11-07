// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>

#include "batch/NetAction.h"
#include "trainers/Trainer.h"

#include "batch/Batcher.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
//#undef PUBLICAPI
//#define PUBLICAPI

/// \brief constructor: pass in data to process, along with labels, network, ...
PUBLICAPI Batcher::Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels) :
        net(net),
        batchSize(batchSize),
        N(N),
        data(data),
        labels(labels)
            {
    inputCubeSize = net->getInputCubeSize();
    numBatches = (N + batchSize - 1) / batchSize;
    reset();
}
VIRTUAL Batcher::~Batcher() {
}
/// \brief reset to the first batch, and set epochDone to false
PUBLICAPI void Batcher::reset() {
    nextBatch = 0;
    numRight = 0;
    loss = 0;
    epochDone = false;
}
/// \brief what is the index of the next batch to process?
PUBLICAPI int Batcher::getNextBatch() {
    if(epochDone) {
        return 0;
    } else {
        return nextBatch;
    }
}
/// \brief for training/testing, what is error loss so far?
PUBLICAPI VIRTUAL float Batcher::getLoss() {
    return loss;
}
/// \brief for training/testing, how many right so far?
PUBLICAPI VIRTUAL int Batcher::getNumRight() {
    return numRight;
}
/// \brief how many examples in the entire set of currently loaded data?
PUBLICAPI VIRTUAL int Batcher::getN() {
    return N;
}
/// \brief has this epoch finished?
PUBLICAPI VIRTUAL bool Batcher::getEpochDone() {
    return epochDone;
}
VIRTUAL void Batcher::setBatchState(int nextBatch, int numRight, float loss) {
    this->nextBatch = nextBatch;
    this->numRight = numRight;
    this->loss = loss;
}
VIRTUAL void Batcher::setN(int N) {
    this->N = N;
    this->numBatches = (N + batchSize - 1) / batchSize;
}
/// \brief processes one single batch of data
///
/// could be learning for one batch, or prediction/testing for one batch
///
/// if most recent epoch has finished, then resets, and starts a new
/// set of learning
PUBLICAPI bool Batcher::tick(int epoch) {
//    cout << "Batcher::tick epochDone=" << epochDone << " batch=" <<  nextBatch << endl;
//    updateVars();
    if(epochDone) {
        reset();
    }
    int batch = nextBatch;
//    std::cout << "BatchLearner.tick() batch=" << batch << std::endl;
    int64 batchStart = batch * batchSize;
    int64 thisBatchSize = batchSize;
    if(batch == numBatches - 1) {
        thisBatchSize = N - batchStart;
    }
//    std::cout << "batchSize=" << batchSize << " thisBatchSize=" << thisBatchSize << " batch=" << batch <<
//            " batchStart=" << batchStart << " data=" << (void *)data << " labels=" << labels << 
//            std::endl;
    net->setBatchSize(thisBatchSize);
    internalTick(epoch, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]));
//        netAction->run(net, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]));
    float thisLoss = net->calcLossFromLabels(&(labels[batchStart]));
    int thisNumRight = net->calcNumRight(&(labels[batchStart]));
//        std::cout << "thisloss " << thisLoss << " thisnumright " << thisNumRight << std::endl; 
    loss += thisLoss;
    numRight += thisNumRight;
    nextBatch++;
    if(nextBatch == numBatches) {
        epochDone = true;
    }
    return !epochDone;
}
/// \brief runs batch once, for currently loaded data
///
/// could be one batch of learning, or one batch of forward propagation
/// (for test/prediction), for example
PUBLICAPI EpochResult Batcher::run(int epoch) {
    if(data == 0) {
        throw runtime_error("Batcher: no data set");
    }
    if(labels == 0) {
        throw runtime_error("Batcher: no labels set");
    }
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}
LearnBatcher::LearnBatcher(Trainer *trainer, Trainable *net,
        int batchSize, int N, float *data, int const*labels) :
    Batcher(net, batchSize, N, data, labels),
    trainer(trainer) {
}
VIRTUAL void LearnBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
//    cout << "LearnBatcher learningRate=" << learningRate << " batchdata=" << (void *)batchData << 
//        " batchLabels=" << batchLabels << endl;
    TrainingContext context(epoch, nextBatch);
    trainer->trainFromLabels(net, &context, batchData, batchLabels);
}

NetActionBatcher::NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction *netAction) :
    Batcher(net, batchSize, N, data, labels),
    netAction(netAction) {
}
void NetActionBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
    netAction->run(this->net, epoch, nextBatch, batchData, batchLabels);
}
ForwardBatcher::ForwardBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels) :
    Batcher(net, batchSize, N, data, labels) {
}
void ForwardBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
    this->net->forward(batchData);
}

