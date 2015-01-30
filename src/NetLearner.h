// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#define VIRTUAL virtual
#define STATIC static

class NeuralNet;

#include "DllImportExport.h"

class ClConvolve_EXPORT PostEpochAction {
public:
    virtual void run( int epoch ) = 0;
};

template<typename T>
class ClConvolve_EXPORT NetLearner {
public:
    NeuralNet *net;

    int Ntrain;
    int Ntest;
    T *trainData;
    int *trainLabels;
    T *testData;
    int *testLabels;

    int batchSize;

    float learningRate;
    float annealLearningRate;

    int startEpoch;
    int numEpochs;

    float translate;
    float scale;

    std::vector<PostEpochAction *> postEpochActions;

//    NetLearner( NeuralNet *net );

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    NetLearner( NeuralNet *net );
    void setTrainingData( int Ntrain, T *trainData, int *trainLabels );
    void setTestingData( int Ntest, T *testData, int *testLabels );
    void setSchedule( int numEpochs );
    void setSchedule( int numEpochs, int startEpoch );
    void setNormalize( float translate, float scale );
    void setBatchSize( int batchSize );
    VIRTUAL ~NetLearner();
    VIRTUAL void addPostEpochAction( PostEpochAction *action );
    void learn( float learningRate );
    void learn( float learningRate, float annealLearningRate );

    // [[[end]]]
};


