// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "GenericLoader.h"
#include "NetAction.h"

#include "OnDemandBatcher.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

OnDemandBatcher::OnDemandBatcher( Trainable *net, NetAction *netAction, 
            std::string filepath, int N, int fileReadBatches, int batchSize ) :
            net( net ),
            batchLearner( net ),
            netAction( netAction ),
            filepath( filepath ),
            N( N ),
            fileReadBatches( fileReadBatches ),
            batchSize( batchSize ),
            fileBatchSize( batchSize * fileReadBatches ),
            inputCubeSize( net->getInputCubeSize() )
        {
    dataBuffer = 0;
    labelsBuffer = 0;
    allocatedSize = 0;
//    dataBuffer = new float[ fileBatchSize * inputCubeSize ];
//    labelsBuffer = new int[ fileBatchSize * inputCubeSize ];
//    updateVars();
    reset();
}
VIRTUAL OnDemandBatcher::~OnDemandBatcher() {
    if( dataBuffer != 0 ) {
        delete[] dataBuffer;
    }
    if( labelsBuffer != 0 ) {
        delete[] labelsBuffer;
    }
}
void OnDemandBatcher::updateBuffers() {
    fileBatchSize = batchSize * fileReadBatches;
    fileBatchSize = fileBatchSize > N ? N : fileBatchSize;
    if( fileBatchSize == 0 ) {
        return;
    }
    numFileBatches = ( N + fileBatchSize - 1 ) / fileBatchSize;
    int newAllocatedSize = fileBatchSize;
    if( newAllocatedSize != allocatedSize ) {
        if( dataBuffer != 0 ) {
            delete[] dataBuffer;
        }
        if( labelsBuffer != 0 ) {
            delete[] labelsBuffer;
        }
        dataBuffer = new float[ newAllocatedSize * inputCubeSize ];
        labelsBuffer = new int[ newAllocatedSize ];
        allocatedSize = newAllocatedSize;
    }
}
void OnDemandBatcher::reset() {
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
bool OnDemandBatcher::tick() {
    updateBuffers();
    if( epochDone ) {
        reset();
    }
    int fileBatch = nextFileBatch;
    int fileBatchStart = fileBatch * fileBatchSize;
    int thisFileBatchSize = fileBatchSize;
    if( fileBatch == numFileBatches - 1 ) {
        thisFileBatchSize = N - fileBatchStart;
    }
    cout << "batchlearnerondemand, read data... filebatchstart=" << fileBatchStart << " filebatchsize=" << thisFileBatchSize << endl;
    GenericLoader::load( filepath, dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize );
    EpochResult epochResult = batchLearner.batchedNetAction( batchSize, thisFileBatchSize, dataBuffer, labelsBuffer, netAction );
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if( nextFileBatch == numFileBatches ) {
        epochDone = true;
    }
    return !epochDone;
}
EpochResult OnDemandBatcher::run() {
    if( epochDone ) {
        reset();
    }
    while( !epochDone ) {
        tick();
    }
    EpochResult epochResult( loss, numRight );
    return epochResult;
}


