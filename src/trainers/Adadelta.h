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

class AdadeltaState;
class CLWrapper;
class EasyCL;
class OutputData;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// based on http://arxiv.org/pdf/1212.5701v1.pdf , page 3, Algorithm 1
class DeepCL_EXPORT Adadelta : public Trainer{
public:
    float decay;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Adadelta();
    VIRTUAL std::string asString();
    VIRTUAL void updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
    AdadeltaState *trainerState);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);
    STATIC Adadelta *instance(EasyCL *cl, float decay);
    Adadelta(EasyCL *cl, float decay);

    // [[[end]]]
};

