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

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT WeightsInitializer {
public:
    virtual void initializeWeights(int numWeights, float *weights, int fanin) = 0;
    virtual void initializeBias(int numBias, float *bias, int fanin) = 0;
    virtual ~WeightsInitializer() {
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:

    // [[[end]]]
};

