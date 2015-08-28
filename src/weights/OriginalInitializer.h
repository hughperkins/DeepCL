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

// the original hacky weights initializer, to avoid having to bump
// the major version
class DeepCL_EXPORT OriginalInitializer : public WeightsInitializer {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void initializeWeights(int numWeights, float *weights, int fanin);
    VIRTUAL void initializeBias(int numBias, float *bias, int fanin);

    // [[[end]]]
};

