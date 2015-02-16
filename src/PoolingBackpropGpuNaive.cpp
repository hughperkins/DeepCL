// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "PoolingBackprop.h"
#include "StatefulTimer.h"

#include "PoolingBackpropGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL void PoolingBackpropGpuNaive::backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("PoolingBackpropGpuNaive::backpropErrors start" );

    
    
    StatefulTimer::instance()->timeCheck("PoolingBackpropGpuNaive::backpropErrors end" );
}
PoolingBackpropGpuNaive::PoolingBackpropGpuNaive( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) :
        PoolingBackprop( cl, padZeros, numPlanes, inputBoardSize, poolingSize ) {
    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/PoolingBackpropGpuNaive.cl", "backprop_errors", 'options' )
    // ]]]
    // generated using cog:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// inplane and outplane are always identical, 1:1 mapping, so can just write `plane`\n" 
    "// errors: [n][plane][outrow][outcol]\n" 
    "// selectors: [n][plane][outrow][outcol]\n" 
    "// errorsForUpstream: [n][plane][inrow][incol]\n" 
    "// wont use workgroups (since 'naive')\n" 
    "// one thread per: [n][plane][outrow][outcol]\n" 
    "// globalId: [n][plane][outrow][outcol]\n" 
    "kernel void backprop_errors( const int batchSize,\n" 
    "    global const float *errors, global const int *selectors, global float *errorsForUpstream ) {\n" 
    "//    const int globalId = get_global_id(0);\n" 
    "\n" 
    "\n" 
    "    memset( errorsForUpstream, 0, sizeof( float ) * getInputSize( batchSize ) );\n" 
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        for( int plane = 0; plane < numPlanes; plane++ ) {\n" 
    "            for( int outputRow = 0; outputRow < outputBoardSize; outputRow++ ) {\n" 
    "                int inputRow = outputRow * poolingSize;\n" 
    "                for( int outputCol = 0; outputCol < outputBoardSize; outputCol++ ) {\n" 
    "                    int inputCol = outputCol * poolingSize;\n" 
    "                    int resultIndex = getResultIndex( n, plane, outputRow, outputCol );\n" 
    "                    float error = errors[resultIndex];\n" 
    "                    int selector = selectors[resultIndex];\n" 
    "                    int drow = selector / poolingSize;\n" 
    "                    int dcol = selector % poolingSize;\n" 
    "                    int inputIndex = getInputIndex( n, plane, inputRow + drow, inputCol + dcol );\n" 
    "                    errorsForUpstream[ inputIndex ] = error;\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backprop_errors", options, "cl/PoolingBackpropGpuNaive.cl" );
    // [[[end]]]
);
}

