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

class AdagradState;
class CLWrapper;
class EasyCL;
class OutputData;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// based on http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
// and http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
class DeepCL_EXPORT Adagrad : public Trainer{
public:
    float fudgeFactor; // if you have a better name, let me know :-)

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Adagrad();
    VIRTUAL void setFudgeFactor(float fudgeFactor);
    VIRTUAL std::string asString();
    VIRTUAL void updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
    AdagradState *trainerState);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);
    STATIC Adagrad *instance(EasyCL *cl, float learningRate);
    Adagrad(EasyCL *cl);

    // [[[end]]]
};

