// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "conv/Forward.h"

class AddBias;
class ReduceSegments;

#define STATIC static
#define VIRTUAL virtual

class ForwardFc : public Forward {
public:
    easycl::CLKernel *kernel1;
//    CLKernel *kernel_reduce;
    AddBias *addBias;
    ReduceSegments *reduceSegments;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardFc();
    VIRTUAL void forward(int batchSize, easycl::CLWrapper *dataWrapper, easycl::CLWrapper *weightsWrapper, easycl::CLWrapper *biasWrapper, easycl::CLWrapper *outputWrapper);
    ForwardFc(easycl::EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

