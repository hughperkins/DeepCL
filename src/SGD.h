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

#include "Trainer.h"

class SGDState;
class CLWrapper;
class OpenCLHelper;
class CLKernel;

#define VIRTUAL virtual
#define STATIC static

class SGD : public Trainer{
public:
    float momentum;

    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~SGD();
    VIRTUAL void setMomentum( float momentum );
    VIRTUAL std::string asString();
    VIRTUAL void updateWeights( CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
    SGDState *trainerState );
    VIRTUAL void train( NeuralNet *net, float const*input, float const*expectedOutput );
    VIRTUAL void trainFromLabels( NeuralNet *net, float const*input, int const*labels );
    VIRTUAL void bindState( NeuralNet *net );
    STATIC SGD *instance( OpenCLHelper *cl, float learningRate, float momentum );
    SGD( OpenCLHelper *cl );

    // [[[end]]]
};

