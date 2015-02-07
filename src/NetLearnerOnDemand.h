// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#define VIRTUAL virtual
#define STATIC static

#include "NetLearner.h"

class NeuralNet;
class Trainable;

#include "DllImportExport.h"

//class ClConvolve_EXPORT PostEpochAction {
//public:
//    virtual void run( int epoch ) = 0;
//};

// handles learning the neural net, ie running multiple epochs,
// using a BatchLearner, to learn each epoch
template<typename T>
class ClConvolve_EXPORT NetLearnerOnDemand {
public:
    Trainable *net;

    int Ntrain;
    int Ntest;
    std::string trainFilepath;
    std::string testFilepath;

//    T *trainData;
//    int *trainLabels;
//    T *testData;
//    int *testLabels;

    int batchSize;

    float learningRate;
    float annealLearningRate;

    bool dumpTimings;

    int startEpoch;
    int numEpochs;

    std::vector<PostEpochAction *> postEpochActions;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    NetLearnerOnDemand( Trainable *net );
    void setTrainingData( std::string trainFilepath, int Ntrain );
    void setTestingData( std::string testFilepath, int Ntest );
    void setSchedule( int numEpochs );
    void setDumpTimings( bool dumpTimings );
    void setSchedule( int numEpochs, int startEpoch );
    void setBatchSize( int batchSize );
    VIRTUAL ~NetLearnerOnDemand();
    VIRTUAL void addPostEpochAction( PostEpochAction *action );
    void learn( float learningRate );
    void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


