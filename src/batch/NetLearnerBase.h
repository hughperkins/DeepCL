// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>

#include "DeepCLDllExport.h"

class Trainer;

class DeepCL_EXPORT NetLearnerBase {
public:
    virtual ~NetLearnerBase() {}
    virtual bool isLearningDone() = 0;
    virtual void setSchedule(int numEpochs) = 0;
    virtual void setDumpTimings(bool dumpTimings) = 0;
    virtual void setSchedule(int numEpochs, int startEpoch) = 0;
//    virtual void setLearningRate(float learningRate) = 0;
//    virtual void setLearningRate(float learningRate, float annealLearningRate) = 0;
//    virtual void learn(float learningRate) = 0;
//    virtual void learn(float learningRate, float annealLearningRate) = 0;
    virtual void reset() = 0;
    virtual bool tickEpoch() = 0;
    virtual bool tickBatch() = 0;
    virtual bool getEpochDone() = 0;
    virtual int getNextEpoch() = 0;
    virtual int getNextBatch() = 0;
    virtual int getBatchNumRight() = 0;
    virtual float getBatchLoss() = 0;
    virtual int getNTrain() = 0;
    virtual void setBatchState(int batch, int numRight, float loss) = 0;
    virtual void run() = 0;
//    virtual void setTrainer(Trainer *trainer) = 0;
};

