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

class GpuOp;
class CLFloatBuffer;
class EasyCL;
class CLKernel;

#include "DeepCLDllExport.h"

// wraps a CLFloatWrapper, so we can do maths on it
// like per-element add, inplace scalar multiply etc
// a bit basic for now.  can extend gradually :-)
// something to consider: pros/cons of using eg clBLAS instead?
class DeepCL_EXPORT CLMathWrapper {
    EasyCL *cl; // dont delete
    GpuOp *gpuOp;

    int N;
    CLFloatWrapper *wrapper; // dont delete

public:
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~CLMathWrapper();
    VIRTUAL CLMathWrapper &operator=(const float scalar);
    VIRTUAL CLMathWrapper &operator*=(const float scalar);
    VIRTUAL CLMathWrapper &operator+=(const float scalar);
    VIRTUAL CLMathWrapper &operator*=(const CLMathWrapper &two);
    VIRTUAL CLMathWrapper &operator+=(const CLMathWrapper &two);
    VIRTUAL CLMathWrapper &operator=(const CLMathWrapper &rhs);
    VIRTUAL CLMathWrapper &sqrt();
    VIRTUAL CLMathWrapper &inv();
    VIRTUAL CLMathWrapper &squared();
    VIRTUAL void runKernel(CLKernel *kernel);
    CLMathWrapper(CLWrapper *wrapper);

    // [[[end]]]
};

