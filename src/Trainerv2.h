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
class Trainerv2 {
public:
    OpenCLHelper *cl;
    NeuralNet *net;

    float learningRate;

    virtual void learn( float *input, float *expectedOutput ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Trainerv2( OpenCLHelper *cl, NeuralNet *net );
    VIRTUAL ~Trainerv2();
    VIRTUAL void setLearningRate( float learningRate );
    VIRTUAL std::string asString();

    // [[[end]]]
};

