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

#include "TrainerMaker.h"

#define VIRTUAL virtual
#define STATIC static

class SGDMaker : public TrainerMaker {
public:
    float learningRate;
    float momentum;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    SGDMaker( float learningRate );
    SGDMaker( float learningRate, float momentum );
    Trainer *instance( OpenCLHelper *cl, int numWeights );

    // [[[end]]]
};

