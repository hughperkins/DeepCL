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
class NetForwardAction;
class OnDemandBatcher;
class Trainer;

#include "DeepCLDllExport.h"

/// \brief Learns multiple epochs, for data that wont fit in memory
///
/// Handles learning the neural net, ie running multiple epochs,
/// using two OnDemandBatchers, one for training, one for testing, to learn 
/// the epochs.
///
/// Note that there's no particular reason why this class couldnt be 
/// merged completely with the 'NetLeaner' class, simply passing 
/// in either 'Batcher' objects, or 'OnDemandBatcher' objects
PUBLICAPI
class DeepCL_EXPORT NetLearnerOnDemand : public NetLearnerBase {
protected:
    Timer timer;
    Trainable *net;
    NetLearnLabeledAction *learnAction;
    NetForwardAction *testAction;
    OnDemandBatcher *learnBatcher;
    OnDemandBatcher *testBatcher;
public:

//    float learningRate;
//    float annealLearningRate;

    bool dumpTimings;

    int nextEpoch;
    int numEpochs;
    bool learningDone;

//    Trainer *trainer;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI NetLearnerOnDemand(Trainer *trainer, Trainable *net,
    std::string trainFilepath, int Ntrain,
    std::string testFilepath, int Ntest,
    int fileReadBatches, int batchSize);
    VIRTUAL ~NetLearnerOnDemand();
    VIRTUAL void setSchedule(int numEpochs);
    VIRTUAL void setDumpTimings(bool dumpTimings);
    VIRTUAL void setSchedule(int numEpochs, int nextEpoch);
    PUBLICAPI VIRTUAL bool getEpochDone();
    PUBLICAPI VIRTUAL int getNextEpoch();
    PUBLICAPI VIRTUAL int getNextBatch();
    PUBLICAPI VIRTUAL int getNTrain();
    PUBLICAPI VIRTUAL int getBatchNumRight();
    PUBLICAPI VIRTUAL float getBatchLoss();
    VIRTUAL void setBatchState(int nextBatch, int numRight, float loss);
    PUBLICAPI VIRTUAL void reset();
    VIRTUAL void postEpochTesting();
    PUBLICAPI VIRTUAL bool tickBatch();  // means: filebatch, not low-level batch
    PUBLICAPI VIRTUAL bool tickEpoch();
    PUBLICAPI VIRTUAL void run();
    PUBLICAPI VIRTUAL bool isLearningDone();

    // [[[end]]]
};


