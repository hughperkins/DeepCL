// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "EasyCL.h"

#include "util/StatefulTimer.h"
#include "util/stringhelper.h"
#include "activate/ActivationFunction.h"

#include "activate/ActivationForwardGpuNaive.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL ActivationForwardGpuNaive::~ActivationForwardGpuNaive() {
    delete kernel;
}
VIRTUAL void ActivationForwardGpuNaive::forward( int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
//    cout << StatefulTimer::instance()->prefix << "ActivationForwardGpuNaive::forward( CLWrapper * )" << endl;
    StatefulTimer::instance()->timeCheck("ActivationForwardGpuNaive::forward start" );

    kernel->input( batchSize * numPlanes * outputImageSize * outputImageSize );
    kernel->output( outputWrapper )->input( inputWrapper );
//    kernel->input( batchSize )->input( inputWrapper )->output( outputWrapper );

    int globalSize = batchSize * numPlanes * outputImageSize * outputImageSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    cout << "ActivationForwardGpuNaive::forward batchsize=" << batchSize << " g=" << globalSize << " w=" << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

//    cout << "ActivationForwardGpuNaive::forward selectorswrapper:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );

    StatefulTimer::instance()->timeCheck("ActivationForwardGpuNaive::forward end" );
}
ActivationForwardGpuNaive::ActivationForwardGpuNaive( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        ActivationForward( cl, numPlanes, inputImageSize, fn ) {
    string options = "";
    options += " -DgOutputImageSize=" + toString( outputImageSize );
    options += " -DgOutputImageSizeSquared=" + toString( outputImageSize * outputImageSize );
    options += " -DgInputImageSize=" + toString( inputImageSize );
    options += " -DgInputImageSizeSquared=" + toString( inputImageSize * inputImageSize );
    options += " -DgNumPlanes=" + toString( numPlanes );
    options += " -D" + fn->getDefineName();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/activate.cl", "forwardNaive", 'options' )
    // ]]]
    // generated using cog, from cl/activate.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH ]\n" 
    "\n" 
    "#ifdef TANH\n" 
    "    #define ACTIVATION_FUNCTION(output) (tanh(output))\n" 
    "#elif defined SCALEDTANH\n" 
    "    #define ACTIVATION_FUNCTION(output) ( 1.7159f * tanh( 0.66667f * output))\n" 
    "#elif SIGMOID\n" 
    "    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))\n" 
    "#elif defined RELU\n" 
    "    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)\n" 
    "#elif defined LINEAR\n" 
    "    #define ACTIVATION_FUNCTION(output) (output)\n" 
    "#endif\n" 
    "\n" 
    "#ifdef ACTIVATION_FUNCTION // protect against not defined\n" 
    "kernel void activate( const int N, global float *inout ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    inout[globalId] = ACTIVATION_FUNCTION( inout[globalId] );\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "#ifdef ACTIVATION_FUNCTION // protect against not defined\n" 
    "kernel void forwardNaive( const int N, global float *out, global const float *in ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    out[globalId] = ACTIVATION_FUNCTION( in[globalId] ); // probably not ideal...\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "forwardNaive", options, "cl/activate.cl" );
    // [[[end]]]
}

