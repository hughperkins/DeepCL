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

class EasyCL;
class CLWrapper;
class CLKernel;

#define VIRTUAL virtual
#define STATIC static

// use to update one buffer by adding another buffer, in-element
// not thread-safe
class GpuAdd {
public:
    EasyCL *cl; // NOT belong to us, dont delete
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void add(int N, CLWrapper*destinationWrapper, CLWrapper *deltaWrapper);
    VIRTUAL ~GpuAdd();
    GpuAdd(EasyCL *cl);

    // [[[end]]]
};

