// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "batch/NetAction.h"
#include "net/Trainable.h"
#include "loaders/GenericLoaderv2.h"
#include "batch/Batcher.h"

#include "batch/OnDemandBatcherv2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI OnDemandBatcherv2::OnDemandBatcherv2(Trainable *net, NetAction *netAction, 
            GenericLoaderv2 *loader, int N, int fileReadBatches, int batchSize) :
            net(net),
            netAction(netAction),
            netActionBatcher(0),
            loader(loader),
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
VIRTUAL OnDemandBatcherv2::~OnDemandBatcherv2() {
    delete netActionBatcher;
    delete[] dataBuffer;
    delete[] labelsBuffer;
}
VIRTUAL void OnDemandBatcherv2::setBatchState(int nextBatch, int numRight, float loss) {
    this->nextFileBatch = nextBatch / fileReadBatches;
    this->numRight = numRight;
    this->loss = loss;
    epochDone = false;
}
VIRTUAL int OnDemandBatcherv2::getBatchSize() {
    return batchSize;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNextFileBatch() {
    return nextFileBatch;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNextBatch() {
    return nextFileBatch * fileReadBatches;
}
PUBLICAPI VIRTUAL float OnDemandBatcherv2::getLoss() {
    return loss;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNumRight() {
    return numRight;
}
PUBLICAPI VIRTUAL bool OnDemandBatcherv2::getEpochDone() {
    return epochDone;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getN() {
    return N;
}
//VIRTUAL void OnDemandBatcherv2::setLearningRate(float learningRate) {
//    this->learningRate = learningRate;
//}
//VIRTUAL void OnDemandBatcherv2::setBatchSize(int batchSize) {
//    if(batchSize != this->batchSize) {
//        this->batchSize = batchSize;
////        updateBuffers();
//    }
//}
PUBLICAPI void OnDemandBatcherv2::reset() {
//    cout << "OnDemandBatcherv2::reset()" << endl;
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
PUBLICAPI bool OnDemandBatcherv2::tick(int epoch) {
//    cout << "OnDemandBatcherv2::tick nextFileBatch=" << nextFileBatch << " numRight=" << numRight << 
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
    loader->load(dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize);
    EpochResult epochResult = netActionBatcher->run(epoch);
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if(nextFileBatch == numFileBatches) {
        epochDone = true;
    }
    return !epochDone;
}
PUBLICAPI EpochResult OnDemandBatcherv2::run(int epoch) {
//    cout << "OnDemandBatcherv2::run() epochDone=" << epochDone << endl;
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}

