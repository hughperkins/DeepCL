// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "net/Trainable.h"

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

class LossLayer;

class NeuralNet;

// This handles grouping several NeuralNets into one single MultiNet
class DeepCL_EXPORT MultiNet : public Trainable {
    std::vector<Trainable * > trainables;
    float *output;
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
    MultiNet(int numNets, NeuralNet *model);
    VIRTUAL ~MultiNet();
    VIRTUAL int getInputCubeSize() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL LossLayerMaker *cloneLossLayerMaker() const;
    VIRTUAL float calcLoss(float const *expectedValues);
    VIRTUAL float calcLossFromLabels(int const *labels);
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void setTraining(bool training);
    VIRTUAL int calcNumRight(int const *labels);
    void forwardToOurselves();
    VIRTUAL void forward(float const*images);
    VIRTUAL void backwardFromLabels(int const *labels);
    VIRTUAL void backward(float const *expectedOutput);
    VIRTUAL float const *getOutput() const;
    VIRTUAL int getNumNets() const;
    VIRTUAL Trainable *getNet(int idx);

    // [[[end]]]
};

