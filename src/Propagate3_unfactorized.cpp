// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "Propagate3_unfactorized.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Propagate3_unfactorized::~Propagate3_unfactorized() {
    delete kernel;
//    delete repeatedAdd;
//    delete activate;
}
VIRTUAL void Propagate3_unfactorized::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    StatefulTimer::timeCheck("Propagate3_unfactorized::propagate begin");
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
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

    StatefulTimer::timeCheck("Propagate3_unfactorized::propagate after call propagate");
}
Propagate3_unfactorized::Propagate3_unfactorized( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {

    if( square( dim.outputBoardSize ) > cl->getMaxWorkgroupSize() ) {
        throw runtime_error("cannot use propagate3, since outputboardsize * outputboardsize > maxworkgroupsize");
    }

    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/propagate3_unfactorized.cl", "propagate", 'options' )
    // ]]]
    // generated using cog:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// unfactorized version of propagate3, so can compare performance\n" 
    "\n" 
    "// expects:\n" 
    "// TANH | SCALEDTANH | SIGMOID | RELU | LINEAR\n" 
    "// BIASED (or not)\n" 
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
    "// concept: each workgroup handles convolving one input example with one filtercube\n" 
    "// and writing out one single output plane\n" 
    "//\n" 
    "// workgroup id organized like: [imageid][outplane]\n" 
    "// local id organized like: [outrow][outcol]\n" 
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n" 
    "// number workgroups = 32\n" 
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
    "// results are organized like [imageid][filterid][row][col]\n" 
    "#ifdef ACTIVATION_FUNCTION\n" 
    "void kernel propagate( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "    global float *results,\n" 
    "    #ifdef BIASED\n" 
    "        global const float*biases,\n" 
    "    #endif\n" 
    "    local float *_upstreamBoard, local float *_filterCube ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
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
    "    const int maxu = gPadZeros ? min( gHalfFilterSize - gEven, gOutputBoardSize - 1 - outputRow  - gEven) : gHalfFilterSize - gEven;\n" 
    "    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;\n" 
    "    const int maxv = gPadZeros ? min( gHalfFilterSize - gEven, gOutputBoardSize - 1 - outputCol - gEven) : gHalfFilterSize - gEven;\n" 
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
    "        for( int u = minu; u <= maxu; u++ ) {\n" 
    "            int inputRow = outputRow + u;\n" 
    "            #if gPadZeros == 0\n" 
    "                inputRow += gHalfFilterSize;\n" 
    "            #endif\n" 
    "            int inputboardrowoffset = inputRow * gInputBoardSize;\n" 
    "            int filterrowoffset = filterBoardOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "            for( int v = minv; v <= maxv; v++ ) {\n" 
    "                int inputCol = outputCol + v;\n" 
    "                #if gPadZeros == 0\n" 
    "                    inputCol += gHalfFilterSize;\n" 
    "                #endif\n" 
    "                if( localId < gOutputBoardSizeSquared ) {\n" 
    "                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "\n" 
    "    #ifdef BIASED\n" 
    "        sum += biases[outPlane];\n" 
    "    #endif\n" 
    "\n" 
    "    // results are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + localId;\n" 
    "    if( localId < gOutputBoardSizeSquared ) {\n" 
    "        results[resultIndex ] = ACTIVATION_FUNCTION( sum );\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagate", options, "cl/propagate3_unfactorized.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "propagate3.cl", "propagate_3_by_n_outplane", options );

//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}

