// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "BackpropWeights2.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropWeights2Cpu : public BackpropWeights2 {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropWeights2Cpu
    // cppfile: BackpropWeights2Cpu.cpp

    BackpropWeights2Cpu( OpenCLHelper *cl, LayerDimensions dim );
    VIRTUAL ~BackpropWeights2Cpu();
    VIRTUAL void backpropWeights( int batchSize, float learningRate,  CLWrapper *derivLossBySumWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper );
    VIRTUAL void backpropWeights( int batchSize, float learningRate, float *derivLossBySum,
    float *images, float *weights, float *biasWeights );

    // [[[end]]]
};

