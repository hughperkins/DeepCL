// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "GenericLoader.h"
#include "NetAction.h"
#include "Trainable.h"
#include "Batcher.h"

#include "OnDemandBatcher.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

OnDemandBatcher::OnDemandBatcher( Trainable *net, NetAction *netAction, 
            std::string filepath, int N, int fileReadBatches, int batchSize ) :
            net( net ),
            netAction( netAction ),
            netActionBatcher( 0 ),
            filepath( filepath ),
            N( N ),
            fileReadBatches( fileReadBatches ),
            batchSize( batchSize ),
            fileBatchSize( batchSize * fileReadBatches ),
            inputCubeSize( net->getInputCubeSize() )
        {
    numFileBatches = ( N + fileBatchSize - 1 ) / fileBatchSize;
//    dataBuffer = 0;
//    labelsBuffer = 0;
//    allocatedSize = 0;
    cout << "OnDemandBatcher::OnDemandBatcher inputCubeSize " << inputCubeSize << " filebatchsize " << fileBatchSize << endl;
    dataBuffer = new float[ fileBatchSize * inputCubeSize ];
    labelsBuffer = new int[ fileBatchSize ];
//    int lastFileBatchSize = N - ( numFileBatches - 1 ) * fileBatchSize;
    netActionBatcher = new NetActionBatcher( net, batchSize, fileBatchSize, dataBuffer, labelsBuffer, netAction );
//    updateVars();
    reset();
}
VIRTUAL OnDemandBatcher::~OnDemandBatcher() {
//    delete netActionLastBatcher;
    delete netActionBatcher;
    delete[] dataBuffer;
    delete[] labelsBuffer;
}
//void OnDemandBatcher::updateBuffers() {
//    fileBatchSize = batchSize * fileReadBatches;
//    fileBatchSize = fileBatchSize > N ? N : fileBatchSize;
//    if( fileBatchSize == 0 ) {
//        return;
//    }
//    numFileBatches = ( N + fileBatchSize - 1 ) / fileBatchSize;
//    int newAllocatedSize = fileBatchSize;
//    if( newAllocatedSize != allocatedSize ) {
//        if( dataBuffer != 0 ) {
//            delete[] dataBuffer;
//        }
//        if( labelsBuffer != 0 ) {
//            delete[] labelsBuffer;
//        }
//        dataBuffer = new float[ newAllocatedSize * inputCubeSize ];
//        labelsBuffer = new int[ newAllocatedSize ];
//        netActionBatcher->setData( dataBuffer, labelsBuffer );
////        netActionBatcher->labels = labelsBuffer;
//        allocatedSize = newAllocatedSize;
//    }
//}
VIRTUAL void OnDemandBatcher::setBatchState( int nextFileBatch, int numRight, float loss ) {
    this->nextFileBatch = nextFileBatch;
    this->numRight = numRight;
    this->loss = loss;
    epochDone = false;
}
//VIRTUAL void OnDemandBatcher::setData( std::string filepath ) {
//    this->filepath = filepath;
//}
//VIRTUAL void OnDemandBatcher::setN( int N ) {
//    if( N != this->N ) {
//        this->N = N;
////        updateBuffers();
//    }
//}
VIRTUAL int OnDemandBatcher::getBatchSize() {
    return batchSize;
}
VIRTUAL int OnDemandBatcher::getNextFileBatch() {
    return nextFileBatch;
}
VIRTUAL float OnDemandBatcher::getLoss() {
    return loss;
}
VIRTUAL float OnDemandBatcher::getNumRight() {
    return numRight;
}
VIRTUAL bool OnDemandBatcher::getEpochDone() {
    return epochDone;
}
VIRTUAL int OnDemandBatcher::getN() {
    return N;
}
//VIRTUAL void OnDemandBatcher::setLearningRate( float learningRate ) {
//    this->learningRate = learningRate;
//}
//VIRTUAL void OnDemandBatcher::setBatchSize( int batchSize ) {
//    if( batchSize != this->batchSize ) {
//        this->batchSize = batchSize;
////        updateBuffers();
//    }
//}
void OnDemandBatcher::reset() {
    cout << "OnDemandBatcher::reset()" << endl;
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
bool OnDemandBatcher::tick() {
    cout << "OnDemandBatcher::tick nextFileBatch=" << nextFileBatch << " numRight=" << numRight << 
        " loss=" << loss << " epochDone=" << epochDone << endl;
//    updateBuffers();
    if( epochDone ) {
        reset();
    }
    int fileBatch = nextFileBatch;
    int fileBatchStart = fileBatch * fileBatchSize;
    int thisFileBatchSize = fileBatchSize;
    if( fileBatch == numFileBatches - 1 ) {
        thisFileBatchSize = N - fileBatchStart;
    }
    netActionBatcher->setN( thisFileBatchSize );
    cout << "batchlearnerondemand, read data... filebatchstart=" << fileBatchStart << " filebatchsize=" << thisFileBatchSize << endl;
    GenericLoader::load( filepath, dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize );
    EpochResult epochResult = netActionBatcher->run();
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if( nextFileBatch == numFileBatches ) {
        epochDone = true;
    }
    return !epochDone;
}
EpochResult OnDemandBatcher::run() {
    cout << "OnDemandBatcher::run() epochDone=" << epochDone << endl;
    if( epochDone ) {
        reset();
    }
    while( !epochDone ) {
        tick();
    }
    EpochResult epochResult( loss, numRight );
    return epochResult;
}

