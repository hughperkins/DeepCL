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

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

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

    virtual void train( NeuralNet *net, float const*input, float const*expectedOutput ) = 0;
    virtual void trainFromLabels( NeuralNet *net, float const*input, int const*labels ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Trainer( EasyCL *cl );
    VIRTUAL ~Trainer();
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL std::string asString();
    VIRTUAL void train( Trainable *trainable, float const*input, float const*expectedOutput );
    VIRTUAL void trainFromLabels( Trainable *trainable, float const*input, int const*labels );

    // [[[end]]]
};

