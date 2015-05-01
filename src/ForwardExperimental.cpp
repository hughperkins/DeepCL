// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ForwardExperimental.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardExperimental::~ForwardExperimental() {
    delete kernel;
}
VIRTUAL void ForwardExperimental::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("ForwardExperimental::forward start");
//    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
//    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
//    if( dim.biased ) kernel->input( biasWrapper );
    kernel->output( outputWrapper );
//    cout << "square(dim.outputImageSize) " << square( dim.outputImageSize ) << endl;
    kernel->localFloats( square( dim.inputImageSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );

    int workgroupsize = std::max( 32, square( dim.outputImageSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "forward3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();

    StatefulTimer::timeCheck("ForwardExperimental::forward after call forward");
}
ForwardExperimental::ForwardExperimental( EasyCL *cl, LayerDimensions dim ) :
        Forward( cl, dim )
            {
    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
    // [[[cog
    // import stringify
    // import cog_getfilename
    // classname = cog_getfilename.get_class_name()
    // stringify.write_kernel2( "kernel", "cl/" + classname + ".cl", "forward", 'options' )
    // ]]]
    // generated using cog, from cl/ForwardExperimental.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// from forward3.cl\n" 
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
    "void kernel forward( const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
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
    "\n" 
    "    // output are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared + localId;\n" 
    "    if( localId < gOutputImageSizeSquared ) {\n" 
    "        output[resultIndex ] = sum;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "forward", options, "cl/ForwardExperimental.cl" );
    // [[[end]]]
}

