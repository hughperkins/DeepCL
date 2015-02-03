// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "DllImportExport.h"

// This handles grouping several NeuralNets into one single MultiNet
class ClConvolve_EXPORT ITrainable {
public:
    virtual float calcLoss(float const *expectedValues ) = 0;
    virtual float calcLossFromLabels(int const *labels ) = 0;
    virtual void setBatchSize( int batchSize ) = 0;
    virtual void setTraining( bool training ) = 0;
    virtual float doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels ) = 0;
    virtual float doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels, int *p_totalCorrect ) = 0;
    virtual float doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults ) = 0;
    virtual int calcNumRight( int const *labels ) = 0;
    virtual template< typename T > void propagate( T const*images) = 0;
    virtual void backPropFromLabels( float learningRate, int const *labels) = 0;
    virtual void backProp( float learningRate, float const *expectedResults) = 0;
    virtual template< typename T > void learnBatch( float learningRate, T const*images, float const *expectedResults ) = 0;
    virtual template< typename T > void learnBatchFromLabels( float learningRate, T const*images, int const *labels ) = 0;
    virtual float const *getResults( int layer ) const = 0;
};

