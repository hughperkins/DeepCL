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

class RmspropState;
class CLWrapper;
class EasyCL;
class OutputData;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// based on http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
// page 29
class DeepCL_EXPORT Rmsprop : public Trainer{
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Rmsprop();
    VIRTUAL std::string asString();
    VIRTUAL void updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
    RmspropState *trainerState);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);
    STATIC Rmsprop *instance(EasyCL *cl, float learningRate);
    Rmsprop(EasyCL *cl);

    // [[[end]]]
};

