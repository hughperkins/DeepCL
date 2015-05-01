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
class EasyCL;
class CLKernel;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT SGD : public Trainer{
public:
    CLKernel *kernel;

    float momentum;

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
    STATIC SGD *instance( EasyCL *cl, float learningRate );
    STATIC SGD *instance( EasyCL *cl, float learningRate, float momentum );
    SGD( EasyCL *cl );

    // [[[end]]]
};

