// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#include "BatchLearner.h"
#include "NetLearnerBase.h"
#include "Trainable.h"
#include "Timer.h"

#define VIRTUAL virtual
#define STATIC static

class NeuralNet;
//class Trainable;

#include "DeepCLDllExport.h"

// handles learning the neural net, ie running multiple epochs,
// using a BatchLearner, to learn each epoch
class DeepCL_EXPORT NetLearner : public NetLearnerBase {
public:
    Trainable *net;
    LearnBatcher *trainBatcher;
    PropagateBatcher *testBatcher;

    float learningRate;
    float annealLearningRate;
//    float annealedLearningRate;

    bool dumpTimings;

    Timer timer;
    int numEpochs;
    int nextEpoch;
    bool learningDone;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NetLearner( Trainable *net );
    VIRTUAL ~NetLearner();
    VIRTUAL void setTrainingData( int Ntrain, float *trainData, int *trainLabels );
    VIRTUAL void setTestingData( int Ntest, float *testData, int *testLabels );
    VIRTUAL void setSchedule( int numEpochs );
    VIRTUAL void setDumpTimings( bool dumpTimings );
    VIRTUAL void setSchedule( int numEpochs, int nextEpoch );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void reset();
    VIRTUAL bool tickEpoch();
    VIRTUAL void run();
    VIRTUAL bool isLearningDone();
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL void setLearningRate( float learningRate, float annealLearningRate );
    VIRTUAL void learn( float learningRate );
    VIRTUAL void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


