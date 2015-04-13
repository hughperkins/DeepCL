// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#include "BatchLearner.h"

#define VIRTUAL virtual
#define STATIC static

class NeuralNet;
class Trainable;

#include "DeepCLDllExport.h"

class DeepCL_EXPORT PostEpochAction {
public:
    virtual void run( int epoch ) = 0;
};

class DeepCL_EXPORT NetLearner_PostBatchAction {
public:
    virtual void run( int epoch, int batch, float lossSoFar, int numRightSoFar ) = 0;
};

class NetLearnerPostBatchRunner : public PostBatchAction {
public:
    int epoch;
    std::vector<NetLearner_PostBatchAction *> postBatchActions; // note: we DONT own these, dont delete, caller owns
    NetLearnerPostBatchRunner() {
        epoch = 0;
    }
    virtual void run( int batch, float lossSoFar, int numRightSoFar ) {
        for( std::vector<NetLearner_PostBatchAction *>::iterator it = postBatchActions.begin(); it != postBatchActions.end(); it++ ) {
            ( *it )->run( epoch, batch, lossSoFar, numRightSoFar );
        }
    }
};

// handles learning the neural net, ie running multiple epochs,
// using a BatchLearner, to learn each epoch
template<typename T>
class DeepCL_EXPORT NetLearner {
public:
    Trainable *net;

    int Ntrain;
    int Ntest;
    T *trainData;
    int *trainLabels;
    T *testData;
    int *testLabels;

    int batchSize;

    float learningRate;
    float annealLearningRate;

    bool dumpTimings;

    int startEpoch;
    int numEpochs;

    std::vector<PostEpochAction *> postEpochActions; // note: we DONT own these, dont delete, caller owns
    std::vector<NetLearner_PostBatchAction *> postBatchActions; // note: we DONT own these, dont delete, caller owns

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    NetLearner( Trainable *net );
    void setTrainingData( int Ntrain, T *trainData, int *trainLabels );
    void setTestingData( int Ntest, T *testData, int *testLabels );
    void setSchedule( int numEpochs );
    void setDumpTimings( bool dumpTimings );
    void setSchedule( int numEpochs, int startEpoch );
    void setBatchSize( int batchSize );
    VIRTUAL ~NetLearner();
    VIRTUAL void addPostEpochAction( PostEpochAction *action );
    VIRTUAL void addPostBatchAction( NetLearner_PostBatchAction *action );
    void learn( float learningRate );
    void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


