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

class NeuralNet;
class Trainable;

#include "DeepCLDllExport.h"

//class DeepCL_EXPORT PostEpochAction {
//public:
//    virtual void run( int epoch ) = 0;
//};

// handles learning the neural net, ie running multiple epochs,
// using a BatchLearner, to learn each epoch
class DeepCL_EXPORT NetLearnerOnDemand : public NetLearnerBase {
public:
    Trainable *net;
    BatchLearnerOnDemand batchLearnerOnDemand;
    Timer timer;

    int Ntrain;
    int Ntest;
    std::string trainFilepath;
    std::string testFilepath;

    int batchSize;
    int fileReadBatches;

    float learningRate;
    float annealLearningRate;

    bool dumpTimings;

    int nextEpoch;
    int numEpochs;
    bool learningDone;

    std::vector<PostEpochAction *> postEpochActions;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NetLearnerOnDemand( Trainable *net );
    VIRTUAL ~NetLearnerOnDemand();
    VIRTUAL void setTrainingData( std::string trainFilepath, int Ntrain );
    VIRTUAL void setTestingData( std::string testFilepath, int Ntest );
    VIRTUAL void setSchedule( int numEpochs );
    VIRTUAL void setDumpTimings( bool dumpTimings );
    VIRTUAL void setSchedule( int numEpochs, int nextEpoch );
    VIRTUAL void setBatchSize( int fileReadBatches, int batchSize );
    VIRTUAL void addPostEpochAction( PostEpochAction *action );
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL void setLearningRate( float learningRate, float annealLearningRate );
    VIRTUAL void reset();
    VIRTUAL bool tickEpoch();
    VIRTUAL void run();
    VIRTUAL bool isLearningDone();
    VIRTUAL void learn( float learningRate );
    VIRTUAL void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


