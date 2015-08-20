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

#include "weights/WeightsInitializer.h"

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// idea of this is that it will assign random floats uniformly sampled
// in range (- multiplier / fanin) to (+ multiplier / fanin)
class DeepCL_EXPORT UniformInitializer : public WeightsInitializer {
public:
    float multiplier;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    UniformInitializer(float multiplier);
    VIRTUAL void initializeWeights(int numWeights, float *weights, int fanin);
    VIRTUAL void initializeBias(int numBias, float *bias, int fanin);

    // [[[end]]]
};

