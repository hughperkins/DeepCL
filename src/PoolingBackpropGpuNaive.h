// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "PoolingBackprop.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingBackpropGpuNaive : public PoolingBackprop {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper,
    CLWrapper *errorsForUpstreamWrapper );
    PoolingBackpropGpuNaive( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize );

    // [[[end]]]
};

