// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Forward.h"

class AddBias;

class Forward1 : public Forward {
public:
    easycl::CLKernel *kernel;
    AddBias *addBias;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward1();
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *dataWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper,
    easycl::CLWrapper *outputWrapper);
    Forward1(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

