// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

//#include "Trainable.h"

// class NeuralNet;
class Trainable;
class EpochResult;
class NetAction;
class Trainer;

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.

class DeepCL_EXPORT BatchLearner {
public:
    Trainable *net; // NOT owned by us, dont delete

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    BatchLearner( Trainable *net );
    EpochResult batchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction );
    EpochResult runBatchedNetAction( int batchSize, int N, float *data, int const*labels, NetAction *netAction );
    int test( int batchSize, int N, float *testData, int const*testLabels );
    int forwardForTrain( int batchSize, int N, float *data, int const*labels );
    EpochResult runEpochFromLabels( Trainer *trainer, int batchSize, int Ntrain, float *trainData, int const*trainLabels );
    float runEpochFromExpected( Trainer *trainer, int batchSize, int N, float *data, float *expectedOutput );

    // [[[end]]]
};

