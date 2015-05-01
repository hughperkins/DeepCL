// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "Forward2.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL Forward2::~Forward2() {
    delete kernel;
}
// only works for small filters
// condition: square( dim.filterSize ) * dim.inputPlanes * 4 < 5000 (about 5KB)
VIRTUAL void Forward2::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper ) {
    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWrapper );
    kernel->output( outputWrapper );
//        cout << "square(outputImageSize) " << square( outputImageSize ) << endl;
    kernel->localFloats( square( dim.inputImageSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );
    int workgroupsize = std::max( 32, square( dim.outputImageSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "forward2 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Forward2::forward after call forward");
}
Forward2::Forward2( EasyCL *cl, LayerDimensions dim ) :
        Forward( cl, dim )
            {
    if( square( dim.outputImageSize ) > cl->getMaxWorkgroupSize() ) {
        throw runtime_error("cannot use forward2, since outputimagesize * outputimagesize > maxworkgroupsize");
    }

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/forward2.cl", "forward_2_by_outplane", 'options' )
    // ]]]
    // generated using cog, from cl/forward2.cl:
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
    "#ifdef gOutputImageSize // for previous tests that dont define it\n" 
    "// workgroup id organized like: [outplane]\n" 
    "// local id organized like: [outrow][outcol]\n" 
    "// each thread iterates over: [imageid][upstreamplane][filterrow][filtercol]\n" 
    "// number workgroups = 32\n" 
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n" 
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n" 
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n" 
    "// output are organized like [imageid][filterid][row][col]\n" 
    "// assumes filter is small, so filtersize * filterSize * inputPlanes * 4 < about 3KB\n" 
    "//                            eg 5 * 5 * 32 * 4 = 3.2KB => ok :-)\n" 
    "//                           but 28 * 28 * 32 * 4 = 100KB => less good :-P\n" 
    "void kernel forward_2_by_outplane( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "        #ifdef BIASED\n" 
    "            global const float*biases,\n" 
    "        #endif\n" 
    "    global float *output,\n" 
    "    local float *_upstreamImage, local float *_filterCube ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "//    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;\n" 
    "\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int outPlane = workgroupId;\n" 
    "\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int outputRow = localId / gOutputImageSize;\n" 
    "    const int outputCol = localId % gOutputImageSize;\n" 
    "\n" 
    "    #if gPadZeros == 1\n" 
    "        const int minu = max( -gHalfFilterSize, -outputRow );\n" 
    "        const int maxu = min( gHalfFilterSize, gOutputImageSize - 1 - outputRow ) - gEven;\n" 
    "        const int minv = max( -gHalfFilterSize, -outputCol );\n" 
    "        const int maxv = min( gHalfFilterSize, gOutputImageSize - 1 - outputCol ) - gEven;\n" 
    "    #else\n" 
    "        const int minu = -gHalfFilterSize;\n" 
    "        const int maxu = gHalfFilterSize - gEven;\n" 
    "        const int minv = -gHalfFilterSize;\n" 
    "        const int maxv = gHalfFilterSize - gEven;\n" 
    "    #endif\n" 
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
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        float sum = 0;\n" 
    "        for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {\n" 
    "            int thisUpstreamImageOffset = ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared;\n" 
    "            barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "            for( int i = 0; i < numUpstreamsPerThread; i++ ) {\n" 
    "                int thisOffset = workgroupSize * i + localId;\n" 
    "                if( thisOffset < gInputImageSizeSquared ) {\n" 
    "                    _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];\n" 
    "                }\n" 
    "            }\n" 
    "            barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "            int filterImageOffset = upstreamPlane * gFilterSizeSquared;\n" 
    "            if( localId < gOutputImageSizeSquared ) {\n" 
    "                for( int u = minu; u <= maxu; u++ ) {\n" 
    "                    int inputRow = outputRow + u;\n" 
    "                    #if gPadZeros == 0\n" 
    "                         inputRow += gHalfFilterSize;\n" 
    "                    #endif\n" 
    "                    int inputimagerowoffset = inputRow * gInputImageSize;\n" 
    "                    int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                    for( int v = minv; v <= maxv; v++ ) {\n" 
    "                        int inputCol = outputCol + v;\n" 
    "                        #if gPadZeros == 0\n" 
    "                             inputCol += gHalfFilterSize;\n" 
    "                        #endif\n" 
    "                        sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                    }\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "        #ifdef BIASED\n" 
    "            sum += biases[outPlane];\n" 
    "        #endif\n" 
    "        // output are organized like [imageid][filterid][row][col]\n" 
    "        int resultIndex = ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared + localId;\n" 
    "        if( localId < gOutputImageSizeSquared ) {\n" 
    "            output[resultIndex ] = sum;\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "forward_2_by_outplane", options, "cl/forward2.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "forward2.cl", "forward_2_by_outplane", options );
}

