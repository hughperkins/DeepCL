// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

class NeuralNet;

#define VIRTUAL virtual
#define STATIC static

#include "DllImportExport.h"

class ClConvolve_EXPORT EpochResult {
public:
    float loss;
    int numRight;
    EpochResult( float loss, int numRight ) :
        loss( loss ),
        numRight( numRight ) {
    }
};

template< typename T>
class ClConvolve_EXPORT NetAction {
public:
    virtual void run( NeuralNet *net, T *batchData, int const*batchLabels ) = 0;
};

template< typename T>
class ClConvolve_EXPORT NetLearnLabeledBatch : public NetAction<T> {
public:
    float learningRate;
    NetLearnLabeledBatch( float learningRate ) :
        learningRate( learningRate ) {
    }
    virtual void run( NeuralNet *net, T *batchData, int const*batchLabels );
};

template< typename T>
class ClConvolve_EXPORT NetPropagateBatch : public NetAction<T> {
public:
    NetPropagateBatch() {
    }
    virtual void run( NeuralNet *net, T *batchData, int const*batchLabels );
};

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.
template< typename T>
class ClConvolve_EXPORT BatchLearner {
public:
    NeuralNet *net; // NOT owned by us, dont delete

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearner( NeuralNet *net );
    EpochResult batchedNetAction( int batchSize, int N, T *data, int const*labels, NetAction<T> *netAction );
    int test( int batchSize, int N, T *testData, int const*testLabels );
    EpochResult runEpochFromLabels( float learningRate, int batchSize, int Ntrain, T *trainData, int const*trainLabels );
    float runEpochFromExpected( float learningRate, int batchSize, int N, T *data, float *expectedResults );

    // [[[end]]]
};

