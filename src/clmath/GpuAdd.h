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

namespace easycl {
class EasyCL;
class CLWrapper;
class CLKernel;
}

#define VIRTUAL virtual
#define STATIC static

// use to update one buffer by adding another buffer, in-element
// not thread-safe
class GpuAdd {
public:
    easycl::EasyCL *cl; // NOT belong to us, dont delete
    easycl::CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void add(int N, easycl::CLWrapper*destinationWrapper, easycl::CLWrapper *deltaWrapper);
    VIRTUAL ~GpuAdd();
    GpuAdd(easycl::EasyCL *cl);

    // [[[end]]]
};

