// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

//#include "Trainable.h"
//#include "BatchLearner.h"

class NetActionBatcher;
class Trainable;
class NetAction;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// This handles an entire epoch of training, loading in data in chunks
// and then passing it to 'Batcher' class, to train/test each chunk
// If you want to run multiple epochs, you'll need a 'NetLearner' class
class OnDemandBatcher {
protected:
//    int allocatedSize;

    Trainable *net;
//    BatchLearner *batchLearner;
    NetAction *netAction; // NOt owned by us, dont delete
    NetActionBatcher *netActionBatcher;
    std::string filepath;
    const int N;
    const int fileReadBatches;
    const int batchSize;
    const int fileBatchSize;
    const int inputCubeSize;
    int numFileBatches;

    float *dataBuffer;
    int *labelsBuffer;

    bool epochDone;
    int numRight;
    float loss;
    int nextFileBatch;

public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    OnDemandBatcher( Trainable *net, NetAction *netAction,
    std::string filepath, int N, int fileReadBatches, int batchSize );
    VIRTUAL ~OnDemandBatcher();
    VIRTUAL void setBatchState( int nextBatch, int numRight, float loss );
    VIRTUAL int getBatchSize();
    VIRTUAL int getNextFileBatch();
    VIRTUAL int getNextBatch();
    VIRTUAL float getLoss();
    VIRTUAL int getNumRight();
    VIRTUAL bool getEpochDone();
    VIRTUAL int getN();
    void reset();
    bool tick();
    EpochResult run();

    // [[[end]]]

//    OnDemandBatcher( Trainable *net, NetAction *netAction, 
//                std::string filepath, int N, int fileReadBatches, int batchSize );
//    virtual ~OnDemandBatcher();
//    void reset();
//    bool tick();
//    EpochResult run();
};

