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
#include "stringhelper.h"

//#include "test/PrintBuffer.h"

#include "PoolingBackpropGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL PoolingBackpropGpuNaive::~PoolingBackpropGpuNaive() {
    delete kernel;
    delete kMemset;
}
VIRTUAL void PoolingBackpropGpuNaive::backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
//    cout << StatefulTimer::instance()->prefix << "PoolingBackpropGpuNaive::backpropErrors( CLWrapper * )" << endl;
//    cout << "PoolingBackpropGpuNaive::backpropErrors" << endl;
    StatefulTimer::instance()->timeCheck("PoolingBackpropGpuNaive::backpropErrors start" );

//    cout << "PoolingBackpropGpuNaive::backpropErrors selectorswrapper:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputBoardSize, outputBoardSize );

    // first, memset errors to 0 ...
    kMemset->out( errorsForUpstreamWrapper )->in( 0.0f )->in( batchSize * numPlanes * inputBoardSize * inputBoardSize );
    int globalSize = batchSize * numPlanes * inputBoardSize * inputBoardSize;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kMemset->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

//    cout << "PoolingBackpropGpuNaive::backpropErrors selectorswrapper after memset...:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputBoardSize, outputBoardSize );

    kernel->in( batchSize )->inout( errorsWrapper )->in( selectorsWrapper )->in( errorsForUpstreamWrapper );
    globalSize = batchSize * numPlanes * outputBoardSize * outputBoardSize;
    workgroupSize = 64;
    numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

//    cout << "PoolingBackpropGpuNaive::backpropErrors selectorswrapper after kernel:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputBoardSize, outputBoardSize );

//    errorsForUpstreamWrapper->copyToHost();
//    float const*errorsForUpstream = reinterpret_cast< float const *>( errorsForUpstreamWrapper->getHostArray() );
//    cout << "inputboardsize " << inputBoardSize << " outputboardsize " << outputBoardSize << endl;
//    for( int i = 0; i < inputBoardSize * inputBoardSize; i++ ) {
////        cout << "errorsForUpstreamWrapper[" << i << "]=" << errorsForUpstream[i] << endl;
//        if( i % inputBoardSize == 0 ) {
//            cout << endl;
//        }
//        cout << errorsForUpstream[i] << " ";
//    }
//            cout << endl;
    
    StatefulTimer::instance()->timeCheck("PoolingBackpropGpuNaive::backpropErrors end" );
}
PoolingBackpropGpuNaive::PoolingBackpropGpuNaive( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) :
        PoolingBackprop( cl, padZeros, numPlanes, inputBoardSize, poolingSize ) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
    options += " -D gNumPlanes=" + toString( numPlanes );
    options += " -D gInputBoardSize=" + toString( inputBoardSize );
    options += " -D gInputBoardSizeSquared=" + toString( inputBoardSize * inputBoardSize );
    options += " -D gOutputBoardSize=" + toString( outputBoardSize );
    options += " -D gOutputBoardSizeSquared=" + toString( outputBoardSize * outputBoardSize );
    options += " -D gPoolingSize=" + toString( poolingSize );
    options += " -D gPadZeros=" + toString( padZeros ? 1 : 0 );

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/PoolingBackpropGpuNaive.cl", "backprop_errors", 'options' )
    // stringify.write_kernel2( "kMemset", "cl/memset.cl", "memset", '""' )
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
    "\n" 
    "    #define globalId get_global_id(0)\n" 
    "    #define nPlaneCombo ( globalId / gOutputBoardSizeSquared )\n" 
    "    #define outputPosCombo ( globalId % gOutputBoardSizeSquared )\n" 
    "\n" 
    "    const int n = nPlaneCombo / gNumPlanes;\n" 
    "    const int plane = nPlaneCombo % gNumPlanes;\n" 
    "    const int outputRow = outputPosCombo / gOutputBoardSize;\n" 
    "    const int outputCol = outputPosCombo % gOutputBoardSize;\n" 
    "\n" 
    "    if( n >= batchSize ) {\n" 
    "        return;\n" 
    "    }\n" 
    "\n" 
    "    int resultIndex = ( ( n\n" 
    "        * gNumPlanes + plane )\n" 
    "        * gOutputBoardSize + outputRow )\n" 
    "        * gOutputBoardSize + outputCol;\n" 
    "    #define error ( errors[resultIndex] )\n" 
    "    int selector = ( selectors[resultIndex] );\n" 
    "    #define drow ( selector / gPoolingSize )\n" 
    "    #define dcol ( selector % gPoolingSize )\n" 
    "    #define inputRow ( outputRow * gPoolingSize + drow )\n" 
    "    #define inputCol ( outputCol * gPoolingSize + dcol )\n" 
    "    int inputIndex = ( ( n\n" 
    "        * gNumPlanes + plane )\n" 
    "        * gInputBoardSize + inputRow )\n" 
    "        * gInputBoardSize + inputCol;\n" 
    "//    if( n < batchSize ) {\n" 
    "        errorsForUpstream[ inputIndex ] = error;\n" 
    "//    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backprop_errors", options, "cl/PoolingBackpropGpuNaive.cl" );
    // generated using cog:
    const char * kMemsetSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void memset( global float *target, const float value, const int N ) {\n" 
    "    #define globalId get_global_id(0)\n" 
    "    if( globalId < N ) {\n" 
    "        target[globalId] = value;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kMemset = cl->buildKernelFromString( kMemsetSource, "memset", "", "cl/memset.cl" );
    // [[[end]]]
}

