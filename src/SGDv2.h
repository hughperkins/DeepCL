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

#include "Trainerv2.h"

#define VIRTUAL virtual
#define STATIC static

class SGDv2 : public Trainerv2 {
public:
    float momentum;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    SGDv2( OpenCLHelper *cl, NeuralNet *net );
    VIRTUAL ~SGDv2();
    VIRTUAL void setMomentum( float momentum );
    VIRTUAL std::string asString();
    VIRTUAL void learn( float *input, float *expectedOutput );  // learns one batch, including updating weights

    // [[[end]]]
};

