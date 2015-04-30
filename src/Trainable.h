// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "DeepCLDllExport.h"

class LossLayerMaker;
class Layer;
class Trainer;
class TrainerMaker;

class DeepCL_EXPORT Trainable {
public:
    virtual ~Trainable() {}
    virtual int getOutputSize() const = 0;
    virtual float calcLoss(float const *expectedValues ) = 0;
    virtual float calcLossFromLabels(int const *labels ) = 0;
    virtual void setBatchSize( int batchSize ) = 0;
    virtual void setTraining( bool training ) = 0;
    virtual int calcNumRight( int const *labels ) = 0;
    virtual void forward( float const*images) = 0;
    virtual void backwardFromLabels( float learningRate, int const *labels) = 0;
    virtual void backward( float learningRate, float const *expectedOutput) = 0;
    virtual float const *getOutput() const = 0;
    virtual LossLayerMaker *cloneLossLayerMaker() const = 0;
    virtual int getOutputPlanes() const = 0;
    virtual int getOutputImageSize() const = 0;
    virtual int getInputCubeSize() const = 0;
    virtual int getOutputCubeSize() const = 0;
//    virtual void setTrainer( TrainerMaker *trainer ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    void learnBatch( float learningRate, float const*images, float const *expectedOutput );
    void learnBatchFromLabels( float learningRate, float const*images, int const *labels );

    // [[[end]]]
};

