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

#include "EasyCL.h"

#define VIRTUAL virtual
#define STATIC static

// adds bias, during forward propagation, after convolutional kernel has run
// but before activation etc
class AddBias {
public:
    EasyCL *cl; // NOT delete
    CLKernel *kernel; // NOT delete

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~AddBias();
    VIRTUAL void forward(
    int batchSize, int numFilters, int outputSize,
    CLWrapper *outputWrapper,
    CLWrapper *biasWrapper
    );
    AddBias(EasyCL *cl);

    // [[[end]]]
};

