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

class LossLayerMaker;
class Layer;

class ClConvolve_EXPORT Trainable {
public:
    virtual int getResultsSize() const = 0;
    virtual float calcLoss(float const *expectedValues ) = 0;
    virtual float calcLossFromLabels(int const *labels ) = 0;
    virtual void setBatchSize( int batchSize ) = 0;
    virtual void setTraining( bool training ) = 0;
    virtual int calcNumRight( int const *labels ) = 0;
    virtual void propagate( float const*images) = 0;
    virtual void propagate( unsigned char const*images) = 0;
    virtual void backPropFromLabels( float learningRate, int const *labels) = 0;
    virtual void backProp( float learningRate, float const *expectedResults) = 0;
    virtual float const *getResults() const = 0;
    virtual LossLayerMaker *cloneLossLayerMaker() const = 0;
    virtual int getOutputPlanes() const = 0;
    virtual int getOutputBoardSize() const = 0;
    virtual int getInputCubeSize() const = 0;
    virtual int getOutputCubeSize() const = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    void learnBatch( float learningRate, float const*images, float const *expectedResults );
    void learnBatch( float learningRate, unsigned char const*images, float const *expectedResults );
    void learnBatchFromLabels( float learningRate, float const*images, int const *labels );
    void learnBatchFromLabels( float learningRate, unsigned char const*images, int const *labels );

    // [[[end]]]
};

