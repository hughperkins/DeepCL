// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "StatefulTimer.h"
#include "EasyCL.h"
#include "GpuAdd.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL void GpuAdd::add( int N, CLWrapper*destinationWrapper, CLWrapper *deltaWrapper ) {
    StatefulTimer::instance()->timeCheck("GpuAdd::add start" );

    kernel->in( N );
    kernel->in( destinationWrapper );
    kernel->in( deltaWrapper );
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuAdd::add end" );
}
VIRTUAL GpuAdd::~GpuAdd() {
}
GpuAdd::GpuAdd( EasyCL *cl ) :
        cl( cl ) {
    static CLKernel *kernel = 0;
    if( kernel != 0 ) {
        this->kernel = kernel;
        return;
    }
    string options = "";

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/per_element_add.cl", "per_element_add", 'options' )
    // ]]]
    // generated using cog, from cl/per_element_add.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void per_element_add( const int N, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[globalId];\n" 
    "}\n" 
    "\n" 
    "// adds source to target\n" 
    "// tiles source as necessary, according to tilingSize\n" 
    "kernel void per_element_tiled_add( const int N, const int tilingSize, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[globalId % tilingSize];\n" 
    "}\n" 
    "\n" 
    "kernel void repeated_add( const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[ ( globalId / repeatSize ) % sourceSize ];\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "per_element_add", options, "cl/per_element_add.cl" );
    // [[[end]]]
    this->kernel = kernel;
}

