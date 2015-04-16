// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#define VIRTUAL virtual
#define STATIC static

#include "NetLearnerBase.h"
#include "NetLearner.h"
#include "BatchLearnerOnDemand.h"
//#include "OnDemandBatcher.h"

class NeuralNet;
class Trainable;
class NetLearnLabeledAction;
class NetPropagateAction;
class OnDemandBatcher;

#include "DeepCLDllExport.h"

//class DeepCL_EXPORT PostEpochAction {
//public:
//    virtual void run( int epoch ) = 0;
//};

// handles learning the neural net, ie running multiple epochs,
// using two OnDemandBatchers, one for training, one for testing, to learn 
// the epochs
// Note that there's no particular reason why this class couldnt be 
// merged completely with the 'NetLeaner' class, simply passing 
// in either 'Batcher' objects, or 'OnDemandBatcher' objects
class DeepCL_EXPORT NetLearnerOnDemand : public NetLearnerBase {
protected:
    Timer timer;
    Trainable *net;
    NetLearnLabeledAction *learnAction;
    NetPropagateAction *testAction;
    OnDemandBatcher *learnBatcher;
    OnDemandBatcher *testBatcher;
public:

//    int Ntrain;
//    int Ntest;
//    std::string trainFilepath;
//    std::string testFilepath;

//    int batchSize;
//    int fileReadBatches;

    float learningRate;
    float annealLearningRate;

    bool dumpTimings;

    int nextEpoch;
    int numEpochs;
    bool learningDone;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NetLearnerOnDemand( Trainable *net,
    std::string trainFilepath, int Ntrain,
    std::string testFilepath, int Ntest,
    int fileReadBatches, int batchSize );
    VIRTUAL ~NetLearnerOnDemand();
    VIRTUAL void setSchedule( int numEpochs );
    VIRTUAL void setDumpTimings( bool dumpTimings );
    VIRTUAL void setSchedule( int numEpochs, int nextEpoch );
    VIRTUAL bool getEpochDone();
    VIRTUAL int getNextEpoch();
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL void setLearningRate( float learningRate, float annealLearningRate );
    VIRTUAL int getNextBatch();
    VIRTUAL int getBatchNumRight();
    VIRTUAL float getBatchLoss();
    VIRTUAL void setBatchState( int nextBatch, int numRight, float loss );
    VIRTUAL void reset();
    VIRTUAL void postEpochTesting();
    VIRTUAL bool tickBatch();  // means: filebatch, not low-level batch
    VIRTUAL bool tickEpoch();
    VIRTUAL void run();
    VIRTUAL bool isLearningDone();
    VIRTUAL void learn( float learningRate );
    VIRTUAL void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


