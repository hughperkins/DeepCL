// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "Propagate3.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Propagate3::~Propagate3() {
    delete kernel;
    delete repeatedAdd;
    delete activate;
}
VIRTUAL void Propagate3::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    StatefulTimer::timeCheck("Propagate3::propagate begin");
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
//    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//    cout << "square(dim.outputBoardSize) " << square( dim.outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );

    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "propagate3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate3::propagate after kernel1");

    if( dim.biased ) {
        repeatedAdd->in( batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize )
            ->in( dim.numFilters )
            ->in( dim.outputBoardSize * dim.outputBoardSize )
            ->inout( resultsWrapper )->in( biasWeightsWrapper );
        maxglobalId = batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize;
        numWorkgroups = ( maxglobalId + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        repeatedAdd->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("Propagate3::propagate after repeatedAdd");
    }

    activate->in( batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize )
        ->inout( resultsWrapper );
    maxglobalId = batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize;
    numWorkgroups = ( maxglobalId + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
    activate->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate3::propagate after activate");

    StatefulTimer::timeCheck("Propagate3::propagate after call propagate");
}
Propagate3::Propagate3( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {

    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/propagate3.cl", "propagate_3_by_n_outplane", 'options' )
    // stringify.write_kernel2( "repeatedAdd", "cl/per_element_add.cl", "repeated_add", 'options' )
    // stringify.write_kernel2( "activate", "cl/activate.cl", "activate", 'options' )
    // ]]]
    // generated using cog:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// BIASED (or not)\n" 
    "\n" 
    "#ifdef gOutputBoardSize // for previous tests that dont define it\n" 
    "// workgroup id organized like: [imageid][outplane]\n" 
    "// local id organized like: [outrow][outcol]\n" 
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n" 
    "// number workgroups = 32\n" 
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
    "// results are organized like [imageid][filterid][row][col]\n" 
    "void kernel propagate_3_by_n_outplane( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "//        #ifdef BIASED\n" 
    "//            global const float*biases,\n" 
    "//        #endif\n" 
    "    global float *results,\n" 
    "    local float *_upstreamBoard, local float *_filterCube ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;\n" 
    "\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int n = workgroupId / gNumFilters;\n" 
    "    const int outPlane = workgroupId % gNumFilters;\n" 
    "\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int outputRow = localId / gOutputBoardSize;\n" 
    "    const int outputCol = localId % gOutputBoardSize;\n" 
    "\n" 
    "    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;\n" 
    "    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;\n" 
    "    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;\n" 
    "    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;\n" 
    "\n" 
    "    const int numUpstreamsPerThread = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "\n" 
    "    const int filterCubeLength = gInputPlanes * gFilterSizeSquared;\n" 
    "    const int filterCubeGlobalOffset = outPlane * filterCubeLength;\n" 
    "    const int numPixelsPerThread = ( filterCubeLength + workgroupSize - 1 ) / workgroupSize;\n" 
    "    for( int i = 0; i < numPixelsPerThread; i++ ) {\n" 
    "        int thisOffset = localId + i * workgroupSize;\n" 
    "        if( thisOffset < filterCubeLength ) {\n" 
    "            _filterCube[thisOffset] = filters[filterCubeGlobalOffset + thisOffset];\n" 
    "        }\n" 
    "    }\n" 
    "    // dont need a barrier, since we'll just run behind the barrier from the upstream board download\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {\n" 
    "        int thisUpstreamBoardOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        for( int i = 0; i < numUpstreamsPerThread; i++ ) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if( thisOffset < gInputBoardSizeSquared ) {\n" 
    "                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];\n" 
    "            }\n" 
    "        }\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        int filterBoardOffset = upstreamPlane * gFilterSizeSquared;\n" 
    "        if( localId < gOutputBoardSizeSquared ) {\n" 
    "            for( int u = minu; u <= maxu; u++ ) {\n" 
    "                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "                int inputboardrowoffset = inputRow * gInputBoardSize;\n" 
    "                int filterrowoffset = filterBoardOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                for( int v = minv; v <= maxv; v++ ) {\n" 
    "                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "//    #ifdef BIASED\n" 
    "//        sum += biases[outPlane];\n" 
    "//    #endif\n" 
    "    // results are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + localId;\n" 
    "    if( localId < gOutputBoardSizeSquared ) {\n" 
    "        results[resultIndex ] = sum;\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagate_3_by_n_outplane", options, "cl/propagate3.cl" );
    // generated using cog:
    const char * repeatedAddSource =  
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
    repeatedAdd = cl->buildKernelFromString( repeatedAddSource, "repeated_add", options, "cl/per_element_add.cl" );
    // generated using cog:
    const char * activateSource =  
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
    "";
    activate = cl->buildKernelFromString( activateSource, "activate", options, "cl/activate.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "propagate3.cl", "propagate_3_by_n_outplane", options );

//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}

