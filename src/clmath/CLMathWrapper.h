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

#define VIRTUAL virtual
#define STATIC static

class CopyBuffer;
class GpuAdd;
class MultiplyInPlace;
class CLFloatBuffer;
class EasyCL;

// wraps a CLFloatWrapper, so we can do maths on it
// like per-element add, inplace scalar multiply etc
class CLMathWrapper {
    EasyCL *cl; // dont delete
    CopyBuffer *copyBuffer;
    GpuAdd *gpuAdd;

    int N;
    CLFloatWrapper *wrapper; // dont delete

public:
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    CLMathWrapper( CLWrapper *wrapper );
    VIRTUAL ~CLMathWrapper();
    VIRTUAL CLMathWrapper &operator+=( const CLMathWrapper &two );
    VIRTUAL CLMathWrapper &operator=( const CLMathWrapper &rhs );

    // [[[end]]]
};

