// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "PoolingPropagate.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingPropagateCpu : public PoolingPropagate {
public:


    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: PoolingPropagateCpu
    // cppfile: PoolingPropagateCpu.cpp

    PoolingPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize );

    // [[[end]]]
};

