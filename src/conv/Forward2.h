// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "conv/Forward.h"

class AddBias;

class Forward2 : public Forward {
public:
    CLKernel *kernel;
    AddBias *addBias;
    int workgroupSize;
    int numWorkgroups;
    int globalSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward2();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper);
    Forward2(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]

};

