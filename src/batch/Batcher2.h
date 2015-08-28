// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "batch/NetAction2.h"
#include "batch/Batcher2.h"
#include "batch/BatchData.h"

#include "DeepCLDllExport.h"

// class NetAction2;
class Trainable;

#define VIRTUAL virtual
#define STATIC static


// concepts behind this batcher:
// - abstract out type of input and output data, via InputData and OutputData object
//   eg can handle both expected outputs, and labeled outputs
// - abstract out action via NetAction2 object
class DeepCL_EXPORT Batcher2 {
protected:
    Trainable *net;
    NetAction2 *action;
    int batchSize;
    int N;
    InputData* inputData;
    OutputData *outputData;

    int numBatches;

    bool epochDone;
    int nextBatch;

public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Batcher2(Trainable *net, NetAction2 *action,
    int batchSize, int N,
    InputData *inputData, OutputData *outputData);
    VIRTUAL ~Batcher2();
    void reset();
    int getNextBatch();
    VIRTUAL int getN();
    VIRTUAL bool getEpochDone();
    VIRTUAL void setN(int N);
    bool tick(int epoch);
    VIRTUAL void internalTick(int epoch, InputData *inputData, OutputData *outputData);
    void run(int epoch);

    // [[[end]]]
};

class DeepCL_EXPORT LearnBatcher2 : public Batcher2 {
public:
    NetLearnAction2 action;

    float epochLoss;
    int epochNumRight;

    LearnBatcher2(Trainable *net, Trainer *trainer, int batchSize, int N,
            InputData *inputData, OutputData *outputData) :
        Batcher2(net, &action, batchSize, N, inputData, outputData),
        action(trainer) {        
    }
    void setBatchState(int nextBatch, int numRight, float loss) {
        this->nextBatch = nextBatch;
        this->epochNumRight = numRight;
        this->epochLoss = loss;
    }
    ~LearnBatcher2() {
    }
    float getEpochLoss() {
        return epochLoss;
    }
    int getEpochNumRight() {
        return epochNumRight;
    }
};


