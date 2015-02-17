// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "CopyBuffer.h"

#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

void CopyBuffer::copy( OpenCLHelper *cl, CLWrapper *sourceWrapper, float *target ) {
    // first we will copy it to another buffer, so we can copy it out
    int bufferSize = sourceWrapper->size();
//    float *copiedBuffer = new float[ bufferSize ];
    CLWrapper *targetWrapper = cl->wrap( bufferSize, target );
    targetWrapper->createOnDevice();

    // now copy it, via a kernel
    const string kernelSource = R"DELIM(
        kernel void copy( int N, global float const *source, global float *dest ) {
            #define globalId ( get_global_id(0) )
            if( globalId < N ) {
                dest[globalId] = source[globalId];
            }
        }
    )DELIM";
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "copy", "" );
    kernel->in( bufferSize )->in( sourceWrapper )->out( targetWrapper );
    int workgroupSize = 32;
    int numWorkgroups = ( bufferSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
    targetWrapper->copyToHost();

    delete targetWrapper;
//    delete[] copiedBuffer;
}

void CopyBuffer::copy( OpenCLHelper *cl, CLWrapper *sourceWrapper, int *target ) {
    // first we will copy it to another buffer, so we can copy it out
    int bufferSize = sourceWrapper->size();
//    float *copiedBuffer = new float[ bufferSize ];
    CLWrapper *targetWrapper = cl->wrap( bufferSize, target );
    targetWrapper->createOnDevice();

    // now copy it, via a kernel
    const string kernelSource = R"DELIM(
        kernel void copy( int N, global int const *source, global int *dest ) {
            #define globalId ( get_global_id(0) )
            if( globalId < N ) {
                dest[globalId] = source[globalId];
            }
        }
    )DELIM";
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "copy", "" );
    kernel->in( bufferSize )->in( sourceWrapper )->out( targetWrapper );
    int workgroupSize = 32;
    int numWorkgroups = ( bufferSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
    targetWrapper->copyToHost();

    delete targetWrapper;
//    delete[] copiedBuffer;
}

