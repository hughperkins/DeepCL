// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

class EasyCL;
class NeuralNet;
class Trainable;
class EpochResult;
class TrainerStateMaker;
class BatchResult;

#include "trainers/TrainingContext.h"

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class BatchResult {
public:
    float loss;
    int numRight;
    BatchResult() {
        loss = 0;
        numRight = 0;
    }
    BatchResult(float loss, int numRight) {
        this->loss = loss;
        this->numRight = numRight;
    }
    float getLoss() {
        return loss;
    }
    int getNumRight() {
        return numRight;
    }
};

// responsible for handling one batch of learning for the passed in network
// TODO: ponder NeuralNet vs Trainable
// Assumptions: this class and its children can assume that the NeuralNet
// is not going to change structure during their lifetime
// If we want to change the NeuralNet structure, we should do it before creating
// the Trainer objects, or we should delete the existing Trainer objects, and
// create new ones
class DeepCL_EXPORT Trainer{
public:
    EasyCL *cl; // NOT delete
//    NeuralNet *net;

    float learningRate;

    virtual BatchResult trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) = 0;
    virtual BatchResult trainNetFromLabels(NeuralNet *net, 
        TrainingContext *context,
        float const*input, int const*labels) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Trainer(EasyCL *cl);
    VIRTUAL ~Trainer();
    VIRTUAL void setLearningRate(float learningRate);
    VIRTUAL std::string asString();
    VIRTUAL BatchResult train(Trainable *trainable,
    TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainFromLabels(Trainable *trainable,
    TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void _bindState(NeuralNet *net, TrainerStateMaker *stateMaker);

    // [[[end]]]
};

