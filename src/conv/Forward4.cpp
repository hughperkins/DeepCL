// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/Forward4.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Forward4::~Forward4() {
    delete kernel;
    delete addBias;
}
VIRTUAL void Forward4::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("Forward4::forward start");

    int numWorkgroups = dim.numFilters * batchSize * pixelsPerThread;
    int globalSize = workgroupSize * numWorkgroups;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->output( outputWrapper );
    kernel->localFloats( square( dim.inputImageSize ) );
    kernel->localFloats( square( dim.filterSize ) );

    kernel->run_1d( globalSize, workgroupSize );
    cl->finish();
    StatefulTimer::timeCheck("Forward4::forward after call forward");

    if( dim.biased ) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputImageSize,
            outputWrapper, biasWrapper );
    }
}
Forward4::Forward4( EasyCL *cl, LayerDimensions dim ) :
        Forward( cl, dim )
            {
    addBias = new AddBias( cl );

    workgroupSize = std::max( 32, square( dim.outputImageSize ) ); // no point in wasting threads....
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    pixelsPerThread = 1;
    while( workgroupSize > maxWorkgroupSize ) {
        workgroupSize >>= 1;
        pixelsPerThread <<= 1;
    }

    std::string options = ""; // "-D " + fn->getDefineName();
    options += " -D gWorkgroupSize=" + toString( workgroupSize );
    options += " -D gPixelsPerThread=" + toString( pixelsPerThread );
    options += dim.buildOptionsString();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/forward4.cl", "forward_4_by_n_outplane_smallercache", 'options' )
    // ]]]
    // generated using cog, from cl/forward4.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "void copyLocal( local float *target, global float const *source, int N ) {\n" 
    "    int numLoops = ( N + get_local_size(0) - 1 ) / get_local_size(0);\n" 
    "    for( int loop = 0; loop < numLoops; loop++ ) {\n" 
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n" 
    "        if( offset < N ) {\n" 
    "            target[offset] = source[offset];\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "#ifdef gOutputImageSize // for previous tests that dont define it\n" 
    "// workgroup id organized like: [n][filterid]\n" 
    "// local id organized like: [outrow][outcol]\n" 
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n" 
    "// number workgroups = 32\n" 
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
    "// output are organized like [n][filterid][outrow][outcol]\n" 
    "void kernel forward_4_by_n_outplane_smallercache( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "    global float *output,\n" 
    "    local float *_upstreamImage, local float *_filterPlane ) {\n" 
    "    #define globalId ( get_global_id(0) )\n" 
    "\n" 
    "    #define localId ( get_local_id(0) )\n" 
    "    #define workgroupId ( get_group_id(0) )\n" 
    "//    const int workgroupSize = get_local_size(0);\n" 
    "    const int effectiveWorkgroupId = workgroupId / gPixelsPerThread;\n" 
    "    const int pixel = workgroupId % gPixelsPerThread;\n" 
    "    const int effectiveLocalId = localId + pixel * gWorkgroupSize;\n" 
    "    const int n = effectiveWorkgroupId / gNumFilters;\n" 
    "    const int outPlane = effectiveWorkgroupId % gNumFilters;\n" 
    "\n" 
    "    const int outputRow = effectiveLocalId / gOutputImageSize;\n" 
    "    const int outputCol = effectiveLocalId % gOutputImageSize;\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        copyLocal( _upstreamImage, images + ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared, gInputImageSizeSquared );\n" 
    "        copyLocal( _filterPlane, filters + ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared, gFilterSizeSquared );\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "\n" 
    "        if( effectiveLocalId < gOutputImageSizeSquared ) {\n" 
    "            for( int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++ ) {\n" 
    "                // trying to reduce register pressure...\n" 
    "                #if gPadZeros == 1\n" 
    "                    #define inputRow ( outputRow + u )\n" 
    "                #else\n" 
    "                    #define inputRow ( outputRow + u + gHalfFilterSize )\n" 
    "                #endif\n" 
    "                int inputimagerowoffset = inputRow * gInputImageSize;\n" 
    "                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                bool rowOk = inputRow >= 0 && inputRow < gInputImageSize;\n" 
    "                for( int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++ ) {\n" 
    "                    #if gPadZeros == 1\n" 
    "                        #define inputCol ( outputCol + v )\n" 
    "                    #else\n" 
    "                        #define inputCol ( outputCol + v + gHalfFilterSize )\n" 
    "                    #endif\n" 
    "                    bool process = rowOk && inputCol >= 0 && inputCol < gInputImageSize;\n" 
    "                    if( process ) {\n" 
    "                            sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterPlane[ filterrowoffset + v ];\n" 
    "                    }\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    // output are organized like [imageid][filterid][row][col]\n" 
    "    #define resultIndex ( ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared + effectiveLocalId )\n" 
    "    if( localId < gOutputImageSizeSquared ) {\n" 
    "        output[resultIndex ] = sum;\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "forward_4_by_n_outplane_smallercache", options, "cl/forward4.cl" );
    // [[[end]]]
}

