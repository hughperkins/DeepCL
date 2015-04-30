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

class OpenCLHelper;
class NeuralNet;

#define VIRTUAL virtual
#define STATIC static

// responsible for handling one batch of learning for the passed in network
// TODO: ponder NeuralNet vs Trainable
// Assumptions: this class and its children can assume that the NeuralNet
// is not going to change structure during their lifetime
// If we want to change the NeuralNet structure, we should do it before creating
// the Trainer objects, or we should delete the existing Trainer objects, and
// create new ones
class Trainer{
public:
    OpenCLHelper *cl; // NOT delete
//    NeuralNet *net;

    float learningRate;

    virtual void train( NeuralNet *net, float *input, float *expectedOutput ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Trainer( OpenCLHelper *cl );
    VIRTUAL ~Trainer();
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL std::string asString();

    // [[[end]]]
};

