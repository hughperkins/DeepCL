// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "Trainable.h"

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

class LossLayer;

class NeuralNet;

// This handles grouping several NeuralNets into one single MultiNet
class DeepCL_EXPORT MultiNet : public Trainable {
    std::vector<Trainable * > trainables;
    float *results;
    int batchSize;
    int allocatedSize;
    InputLayer *proxyInputLayer; // used to feed in output from children, to give to lossLayer
    LossLayer *lossLayer;

public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    MultiNet( int numNets, NeuralNet *model );
    VIRTUAL ~MultiNet();
    VIRTUAL int getInputCubeSize() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getResultsSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL LossLayerMaker *cloneLossLayerMaker() const;
    VIRTUAL float calcLoss(float const *expectedValues );
    VIRTUAL float calcLossFromLabels(int const *labels );
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void setTraining( bool training );
    VIRTUAL int calcNumRight( int const *labels );
    void propagateToOurselves();
    VIRTUAL void propagate( float const*images);
    VIRTUAL void backPropFromLabels( float learningRate, int const *labels);
    VIRTUAL void backProp( float learningRate, float const *expectedResults);
    VIRTUAL float const *getResults() const;

    // [[[end]]]
};

