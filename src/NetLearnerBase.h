// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>

#include "DeepCLDllExport.h"

class DeepCL_EXPORT NetLearnerBase {
public:
    virtual ~NetLearnerBase() {}
    virtual bool isLearningDone() = 0;
    virtual void setSchedule( int numEpochs ) = 0;
    virtual void setDumpTimings( bool dumpTimings ) = 0;
    virtual void setSchedule( int numEpochs, int startEpoch ) = 0;
    virtual void setLearningRate( float learningRate ) = 0;
    virtual void setLearningRate( float learningRate, float annealLearningRate ) = 0;
    virtual void learn( float learningRate ) = 0;
    virtual void learn( float learningRate, float annealLearningRate ) = 0;
    virtual void reset() = 0;
    virtual bool tickEpoch() = 0;
    virtual bool tickBatch() {throw std::runtime_error("tickBatch not implemented");}
    virtual bool isEpochDone() {throw std::runtime_error("isEpochDone not implemented");}
    virtual int getNextEpoch() { throw std::runtime_error("getNextEpoch not implemented");}
    virtual int getNextBatch() { throw std::runtime_error("getNextBatch not implemented");}
    virtual int getBatchNumRight() { throw std::runtime_error("getBatchNumRight not implemented");}
    virtual float getBatchLoss() { throw std::runtime_error("getBatchLoss not implemented");}
    virtual void setBatchState( int batch, int numRight, float loss ) { throw std::runtime_error("setBatchState not implemented");}
    virtual void run() = 0;
};

