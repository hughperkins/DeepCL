// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "PropagateExperimental.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL PropagateExperimental::~PropagateExperimental() {
    delete kernel;
}
VIRTUAL void PropagateExperimental::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    StatefulTimer::timeCheck("PropagateExperimental::propagate start");

    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int pixelsPerThread = 1;
    while( workgroupsize > maxWorkgroupSize ) {
        workgroupsize >>= 1;
        pixelsPerThread <<= 1;
    }
    int numWorkgroups = dim.numFilters * batchSize * pixelsPerThread;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "propagate4 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << " threadsperpixel " << pixelsPerThread << endl;

    kernel->in(batchSize);
    kernel->in( pixelsPerThread );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//    cout << "square(dim.outputBoardSize) " << square( dim.outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) );
//    kernel->localFloats( pixelsPerThread * workgroupsize );

    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("PropagateExperimental::propagate after call propagate");
}
PropagateExperimental::PropagateExperimental( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {
    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
    // [[[cog
    // import stringify
    // import cog_getfilename
    // classname = cog_getfilename.get_class_name()
    // stringify.write_kernel2( "kernel", "cl/" + classname + ".cl", "propagate", 'options' )
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
    "// one of: [ TANH | RELU | LINEAR ]\n" 
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
    "#ifdef gOutputBoardSize // for previous tests that dont define it\n" 
    "#ifdef ACTIVATION_FUNCTION // protect against not defined\n" 
    "// workgroup id organized like: [imageid][outplane]\n" 
    "// local id organized like: [outrow][outcol]\n" 
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n" 
    "// number workgroups = 32\n" 
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
    "// results are organized like [imageid][filterid][row][col]\n" 
    "void kernel propagate( const int batchSize,\n" 
    "    const int pixelsPerThread,\n" 
    "      global const float *images, global const float *filters,\n" 
    "        #ifdef BIASED\n" 
    "            global const float*biases,\n" 
    "        #endif\n" 
    "    global float *results,\n" 
    "    local float *_upstreamBoard, local float *_filterCube ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;\n" 
    "\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int effectiveWorkgroupId = workgroupId / pixelsPerThread;\n" 
    "    const int pixel = workgroupId % pixelsPerThread;\n" 
    "    const int effectiveLocalId = localId + pixel * workgroupSize;\n" 
    "    const int n = effectiveWorkgroupId / gNumFilters;\n" 
    "    const int outPlane = effectiveWorkgroupId % gNumFilters;\n" 
    "\n" 
    "    const int numUpstreamsPerThread = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "    const int numFilterPixelsPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "\n" 
    "    const int outputRow = effectiveLocalId / gOutputBoardSize;\n" 
    "    const int outputCol = effectiveLocalId % gOutputBoardSize;\n" 
    "\n" 
    "    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;\n" 
    "    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;\n" 
    "    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;\n" 
    "    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;\n" 
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
    "        const int filterGlobalOffset = ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared;\n" 
    "        for( int i = 0; i < numFilterPixelsPerThread; i++ ) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if( thisOffset < gFilterSizeSquared ) {\n" 
    "                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];\n" 
    "            }\n" 
    "        }\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "\n" 
    "        if( effectiveLocalId < gOutputBoardSizeSquared ) {\n" 
    "            for( int u = minu; u <= maxu; u++ ) {\n" 
    "                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "                int inputboardrowoffset = inputRow * gInputBoardSize;\n" 
    "                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                for( int v = minv; v <= maxv; v++ ) {\n" 
    "                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    #ifdef BIASED\n" 
    "        sum += biases[outPlane];\n" 
    "    #endif\n" 
    "    // results are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + effectiveLocalId;\n" 
    "    results[resultIndex ] = ACTIVATION_FUNCTION(sum);\n" 
    "    // results[resultIndex ] = 123;\n" 
    "    //if( globalId == 0 ) results[0] += 0.000001f + _perPixelSums[0];\n" 
    "}\n" 
    "#endif\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagate", options, "cl/PropagateExperimental.cl" );
    // [[[end]]]
}

