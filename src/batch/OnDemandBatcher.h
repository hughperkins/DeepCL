// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

//#include "net/Trainable.h"
//#include "BatchLearner.h"

class NetActionBatcher;
class Trainable;
class NetAction;

#include "batch/NetAction.h"

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

/// \brief Learns an entire epoch of training, for data that wont fit in memory
///
/// This handles an entire epoch of training, loading in data in chunks
/// and then passing it to a 'Batcher' class, to train/test each chunk
///
/// If you want to run multiple epochs, you can use a 'NetLearnerOnDemand'
/// class
PUBLICAPI
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
    PUBLICAPI OnDemandBatcher(Trainable *net, NetAction *netAction,
    std::string filepath, int N, int fileReadBatches, int batchSize);
    VIRTUAL ~OnDemandBatcher();
    VIRTUAL void setBatchState(int nextBatch, int numRight, float loss);
    VIRTUAL int getBatchSize();
    PUBLICAPI VIRTUAL int getNextFileBatch();
    PUBLICAPI VIRTUAL int getNextBatch();
    PUBLICAPI VIRTUAL float getLoss();
    PUBLICAPI VIRTUAL int getNumRight();
    PUBLICAPI VIRTUAL bool getEpochDone();
    PUBLICAPI VIRTUAL int getN();
    PUBLICAPI void reset();
    PUBLICAPI bool tick(int epoch);
    PUBLICAPI EpochResult run(int epoch);

    // [[[end]]]
};

