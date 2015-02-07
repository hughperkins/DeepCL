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
class Trainable;

#include "BatchLearner.h"

#define VIRTUAL virtual
#define STATIC static

#include "DllImportExport.h"

//class ClConvolve_EXPORT EpochResult {
//public:
//    float loss;
//    int numRight;
//    EpochResult( float loss, int numRight ) :
//        loss( loss ),
//        numRight( numRight ) {
//    }
//};

//template< typename T>
//class ClConvolve_EXPORT NetAction {
//public:
//    virtual void run( Trainable *net, T *batchData, int const*batchLabels ) = 0;
//};

//template< typename T>
//class ClConvolve_EXPORT NetLearnLabeledBatch : public NetAction<T> {
//public:
//    float learningRate;
//    NetLearnLabeledBatch( float learningRate ) :
//        learningRate( learningRate ) {
//    }
//    virtual void run( Trainable *net, T *batchData, int const*batchLabels );
//};

//template< typename T>
//class ClConvolve_EXPORT NetPropagateBatch : public NetAction<T> {
//public:
//    NetPropagateBatch() {
//    }
//    virtual void run( Trainable *net, T *batchData, int const*batchLabels );
//};

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.
template< typename T>
class ClConvolve_EXPORT BatchLearnerOnDemand {
public:
    Trainable *net; // NOT owned by us, dont delete

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    BatchLearnerOnDemand( Trainable *net );
    EpochResult batchedNetAction( std::string filepath, int batchSize, int N, NetAction<T> *netAction );
    int test( std::string filepath, int batchSize, int Ntest );
    EpochResult runEpochFromLabels( float learningRate, std::string filepath, int batchSize, int Ntrain );

    // [[[end]]]
};

