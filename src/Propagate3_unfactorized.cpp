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
    CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("Propagate3_unfactorized::propagate begin");
//    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
//    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( outputWrapper );
//    cout << "square(dim.outputImageSize) " << square( dim.outputImageSize ) << endl;
    kernel->localFloats( square( dim.inputImageSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );

    int workgroupsize = std::max( 32, square( dim.outputImageSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "propagate3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();

    StatefulTimer::timeCheck("Propagate3_unfactorized::propagate after call propagate");
}
Propagate3_unfactorized::Propagate3_unfactorized( OpenCLHelper *cl, LayerDimensions dim ) :
        Propagate( cl, dim )
            {

    if( square( dim.outputImageSize ) > cl->getMaxWorkgroupSize() ) {
        throw runtime_error("cannot use propagate3, since outputimagesize * outputimagesize > maxworkgroupsize");
    }

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/propagate3_unfactorized.cl", "propagate", 'options' )
    // ]]]
    // generated using cog, from cl/propagate3_unfactorized.cl:
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
    "// BIASED (or not)\n" 
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
    "// output are organized like [imageid][filterid][row][col]\n" 
    "void kernel propagate( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "    #ifdef BIASED\n" 
    "        global const float*biases,\n" 
    "    #endif\n" 
    "    global float *output,\n" 
    "    local float *_upstreamImage, local float *_filterCube ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int n = workgroupId / gNumFilters;\n" 
    "    const int outPlane = workgroupId % gNumFilters;\n" 
    "\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int outputRow = localId / gOutputImageSize;\n" 
    "    const int outputCol = localId % gOutputImageSize;\n" 
    "\n" 
    "    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;\n" 
    "    const int maxu = gPadZeros ? min( gHalfFilterSize - gEven, gOutputImageSize - 1 - outputRow  - gEven) : gHalfFilterSize - gEven;\n" 
    "    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;\n" 
    "    const int maxv = gPadZeros ? min( gHalfFilterSize - gEven, gOutputImageSize - 1 - outputCol - gEven) : gHalfFilterSize - gEven;\n" 
    "\n" 
    "    const int numUpstreamsPerThread = ( gInputImageSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
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
    "    // dont need a barrier, since we'll just run behind the barrier from the upstream image download\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {\n" 
    "        int thisUpstreamImageOffset = ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared;\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        for( int i = 0; i < numUpstreamsPerThread; i++ ) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if( thisOffset < gInputImageSizeSquared ) {\n" 
    "                _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];\n" 
    "            }\n" 
    "        }\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        int filterImageOffset = upstreamPlane * gFilterSizeSquared;\n" 
    "        for( int u = minu; u <= maxu; u++ ) {\n" 
    "            int inputRow = outputRow + u;\n" 
    "            #if gPadZeros == 0\n" 
    "                inputRow += gHalfFilterSize;\n" 
    "            #endif\n" 
    "            int inputimagerowoffset = inputRow * gInputImageSize;\n" 
    "            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "            for( int v = minv; v <= maxv; v++ ) {\n" 
    "                int inputCol = outputCol + v;\n" 
    "                #if gPadZeros == 0\n" 
    "                    inputCol += gHalfFilterSize;\n" 
    "                #endif\n" 
    "                if( localId < gOutputImageSizeSquared ) {\n" 
    "                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "\n" 
    "    #ifdef BIASED\n" 
    "        if( outPlane < gNumFilters ) {\n" 
    "            sum += biases[outPlane];\n" 
    "        }\n" 
    "    #endif\n" 
    "\n" 
    "    // output are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared + localId;\n" 
    "    if( localId < gOutputImageSizeSquared ) {\n" 
    "        output[resultIndex ] = sum;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagate", options, "cl/propagate3_unfactorized.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "propagate3.cl", "propagate_3_by_n_outplane", options );

//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}

