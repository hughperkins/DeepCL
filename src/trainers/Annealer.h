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

#include "trainers/Trainer.h"

class CLWrapper;
class EasyCL;
class NeuralNet;
class OutputData;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// anneals learning, so actual learning rate =
//    learning rate * pow(anneal, epoch)
//    (for zero-based epoch number)
class DeepCL_EXPORT Annealer : public Trainer {
public:
//    CopyBuffer *copyBuffer;
//    GpuAdd *gpuAdd;
//    MultiplyInPlace *multiplyInPlace;

    float anneal;
//    int epoch;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC Annealer *instance(EasyCL *cl, float learningRate, float anneal);
    Annealer(EasyCL *cl);
    VIRTUAL ~Annealer();
    VIRTUAL std::string asString();
    VIRTUAL void setAnneal(float anneal);
    VIRTUAL void updateWeights(float annealedLearningRate, CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper);
    VIRTUAL BatchResult trainNet(
    NeuralNet *net, TrainingContext *context,
    float const *input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);

    // [[[end]]]
};

