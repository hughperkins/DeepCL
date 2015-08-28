// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "loaders/GenericLoader.h"
#include "NetAction.h"
#include "net/Trainable.h"
#include "Batcher.h"

#include "OnDemandBatcher.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI OnDemandBatcher::OnDemandBatcher(Trainable *net, NetAction *netAction, 
            std::string filepath, int N, int fileReadBatches, int batchSize) :
            net(net),
            netAction(netAction),
            netActionBatcher(0),
            filepath(filepath),
            N(N),
            fileReadBatches(fileReadBatches),
            batchSize(batchSize),
            fileBatchSize(batchSize * fileReadBatches),
            inputCubeSize(net->getInputCubeSize())
        {
    numFileBatches = (N + fileBatchSize - 1) / fileBatchSize;
    dataBuffer = new float[ fileBatchSize * inputCubeSize ];
    labelsBuffer = new int[ fileBatchSize ];
    netActionBatcher = new NetActionBatcher(net, batchSize, fileBatchSize, dataBuffer, labelsBuffer, netAction);
    reset();
}
VIRTUAL OnDemandBatcher::~OnDemandBatcher() {
    delete netActionBatcher;
    delete[] dataBuffer;
    delete[] labelsBuffer;
}
VIRTUAL void OnDemandBatcher::setBatchState(int nextBatch, int numRight, float loss) {
    this->nextFileBatch = nextBatch / fileReadBatches;
    this->numRight = numRight;
    this->loss = loss;
    epochDone = false;
}
VIRTUAL int OnDemandBatcher::getBatchSize() {
    return batchSize;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNextFileBatch() {
    return nextFileBatch;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNextBatch() {
    return nextFileBatch * fileReadBatches;
}
PUBLICAPI VIRTUAL float OnDemandBatcher::getLoss() {
    return loss;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNumRight() {
    return numRight;
}
PUBLICAPI VIRTUAL bool OnDemandBatcher::getEpochDone() {
    return epochDone;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getN() {
    return N;
}
//VIRTUAL void OnDemandBatcher::setLearningRate(float learningRate) {
//    this->learningRate = learningRate;
//}
//VIRTUAL void OnDemandBatcher::setBatchSize(int batchSize) {
//    if(batchSize != this->batchSize) {
//        this->batchSize = batchSize;
////        updateBuffers();
//    }
//}
PUBLICAPI void OnDemandBatcher::reset() {
//    cout << "OnDemandBatcher::reset()" << endl;
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
PUBLICAPI bool OnDemandBatcher::tick(int epoch) {
//    cout << "OnDemandBatcher::tick nextFileBatch=" << nextFileBatch << " numRight=" << numRight << 
//        " loss=" << loss << " epochDone=" << epochDone << endl;
//    updateBuffers();
    if(epochDone) {
        reset();
    }
    int fileBatch = nextFileBatch;
    int fileBatchStart = fileBatch * fileBatchSize;
    int thisFileBatchSize = fileBatchSize;
    if(fileBatch == numFileBatches - 1) {
        thisFileBatchSize = N - fileBatchStart;
    }
    netActionBatcher->setN(thisFileBatchSize);
//    cout << "batchlearnerondemand, read data... filebatchstart=" << fileBatchStart << " filebatchsize=" << thisFileBatchSize << endl;
    GenericLoader::load(filepath.c_str(), dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize);
    EpochResult epochResult = netActionBatcher->run(epoch);
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if(nextFileBatch == numFileBatches) {
        epochDone = true;
    }
    return !epochDone;
}
PUBLICAPI EpochResult OnDemandBatcher::run(int epoch) {
//    cout << "OnDemandBatcher::run() epochDone=" << epochDone << endl;
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}

