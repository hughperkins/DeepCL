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

#include "DeepCLDllExport.h"

class NesterovState;

#define VIRTUAL virtual
#define STATIC static

// implements Nesterov momentum
// Nesterov momentum defined eg in http://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf
//      dweights[t+1] = mom * dweights[t] - learningrate * gradient(weights[t] + mom * dweights[t])
//      weights[t+1] = weights[t] + dweights[t+1]
//
// given weights[t], dweights[t]:
//      forward/backprop weights[t] + mom * dweights[t]
//      => calc dweights[t+1]
//      => calc weights[t+1]
//
class DeepCL_EXPORT Nesterov : public Trainer {
public:
    float momentum;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Nesterov();
    VIRTUAL void setMomentum(float momentum);
    VIRTUAL std::string asString();
    VIRTUAL void loadFutureWeights(
    easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *gradWeightsWrapper,
    NesterovState *trainerState);
    VIRTUAL void updateWeights(easycl::CLWrapper *weightsWrapper,
    easycl::CLWrapper *gradWeightsWrapper,
    NesterovState *trainerState);
    VIRTUAL BatchResult trainNet(
    NeuralNet *net, TrainingContext *context,
    float const *input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);
    STATIC Nesterov *instance(easycl::EasyCL *cl, float learningRate);
    STATIC Nesterov *instance(easycl::EasyCL *cl, float learningRate, float momentum);
    Nesterov(easycl::EasyCL *cl);

    // [[[end]]]
};

