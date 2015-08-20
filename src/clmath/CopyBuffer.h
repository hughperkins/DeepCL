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
class CLKernel;
class CLWrapper;

#define VIRTUAL virtual
#define STATIC static

// simply going to copy from one buffer to another
// nothing complicated :-)
class CopyBuffer {
public:
    EasyCL *cl;
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void copy(int N, CLWrapper *in, CLWrapper *out);
    VIRTUAL ~CopyBuffer();
    CopyBuffer(EasyCL *cl);

    // [[[end]]]
};

