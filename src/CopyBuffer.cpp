// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "StatefulTimer.h"
#include "CopyBuffer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL void CopyBuffer::copy( int N, CLWrapper *in, CLWrapper *out ) {
        StatefulTimer::instance()->timeCheck("CopyBuffer::copy start" );

    kernel  ->in( N )
            ->in( in )
            ->out( out );

    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

    StatefulTimer::instance()->timeCheck("CopyBuffer::copy end" );
}

VIRTUAL CopyBuffer::~CopyBuffer() {
    delete kernel;
}

CopyBuffer::CopyBuffer( OpenCLHelper *cl ) :
        cl( cl ) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/copy.cl", "copy", 'options' )
    // ]]]
    // generated using cog, from cl/copy.cl:
    const char * kernelSource =  
    "// simply copies from one to the other...\n" 
    "// there might be something built-in to opencl for this\n" 
    "// anyway... :-)\n" 
    "kernel void copy(\n" 
    "        const int N,\n" 
    "        global const float *in,\n" 
    "        global float *out ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    out[globalId] = in[globalId];\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "copy", options, "cl/copy.cl" );
    // [[[end]]]
}

