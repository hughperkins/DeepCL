// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackpropWeights2Scratch.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeights2Scratch::~BackpropWeights2Scratch() {
    delete kernel;
}
VIRTUAL void BackpropWeights2Scratch::backpropWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeights2Scratch start" );

    int workgroupsize = std::max( 32, square( dim.filterSize ) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

    int localMemRequiredKB = ( square( dim.outputImageSize ) * 4 + square( dim.inputImageSize ) * 4 ) / 1024 + 1;
    if( localMemRequiredKB >= cl->getLocalMemorySizeKB() ) {
        throw runtime_error( "local memory too small to use this kernel on this device.  Need: " + 
            toString( localMemRequiredKB ) + "KB, but only have: " + 
            toString( cl->getLocalMemorySizeKB() ) + "KB local memory" );
    }

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
       ->in( gradOutputWrapper )
        ->in( imagesWrapper )
       ->inout( weightsWrapper );
    if( dim.biased ) {
        kernel->inout( biasWeightsWrapper );
    }
    kernel
        ->localFloats( square( dim.outputImageSize ) )
        ->localFloats( square( dim.inputImageSize ) );

    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeights2Scratch end" );
}
BackpropWeights2Scratch::BackpropWeights2Scratch( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
    std::string options = dim.buildOptionsString();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/BackpropWeights2Scratch.cl", "backprop_floats_withscratch_dobias", 'options' )
    // ]]]
    // generated using cog, from cl/BackpropWeights2Scratch.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// BIASED (or not)\n" 
    "\n" 
    "// including cl/copyLocal.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
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
    "void copyGlobal( global float *target, local float const *source, int N ) {\n" 
    "    int numLoops = ( N + get_local_size(0) - 1 ) / get_local_size(0);\n" 
    "    for( int loop = 0; loop < numLoops; loop++ ) {\n" 
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n" 
    "        if( offset < N ) {\n" 
    "            target[offset] = source[offset];\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "\n" 
    "// including cl/ids.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "#define globalId ( get_global_id(0) )\n" 
    "#define localId ( get_local_id(0)  )\n" 
    "#define workgroupId ( get_group_id(0) )\n" 
    "#define workgroupSize ( get_local_size(0) )\n" 
    "\n" 
    "\n" 
    "\n" 
    "\n" 
    "// workgroupId: [outputPlane][inputPlane]\n" 
    "// localId: [filterRow][filterCol]\n" 
    "// per-thread iteration: [n][outputRow][outputCol]\n" 
    "// local: errorimage: outputImageSize * outputImageSize\n" 
    "//        imageimage: inputImageSize * inputImageSize\n" 
    "void kernel backprop_floats_withscratch_dobias(\n" 
    "        const float learningRateMultiplier, const int batchSize,\n" 
    "         global const float *errors, global const float *images,\n" 
    "        global float *weights,\n" 
    "        #ifdef BIASED\n" 
    "             global float *biasWeights,\n" 
    "        #endif\n" 
    "        local float *_errorImage, local float *_imageImage\n" 
    " ) {\n" 
    "    const int filterRow = localId / gFilterSize;\n" 
    "    const int filterCol = localId % gFilterSize;\n" 
    "\n" 
    "    #define outPlane ( workgroupId / gInputPlanes )\n" 
    "    #define upstreamPlane ( workgroupId % gInputPlanes )\n" 
    "\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "    float thiswchange = 0;\n" 
    "#ifdef BIASED\n" 
    "    float thisbiaschange = 0;\n" 
    "#endif\n" 
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        copyLocal( _imageImage, images + ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared, gInputImageSizeSquared );\n" 
    "        copyLocal(_errorImage, errors + ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared, gOutputImageSizeSquared );\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        if( localId < gFilterSizeSquared ) {\n" 
    "            for( int outRow = 0; outRow < gOutputImageSize; outRow++ ) {\n" 
    "                int upstreamRow = outRow - gMargin + filterRow;\n" 
    "                for( int outCol = 0; outCol < gOutputImageSize; outCol++ ) {\n" 
    "                    const int upstreamCol = outCol - gMargin + filterCol;\n" 
    "                    #define proceed ( upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputImageSize && upstreamCol < gInputImageSize )\n" 
    "                    if( proceed ) {\n" 
    "                        // these defines reduce register pressure, compared to const\n" 
    "                        // giving a 40% speedup on nvidia :-)\n" 
    "                        #define resultIndex ( outRow * gOutputImageSize + outCol )\n" 
    "                        #define error ( _errorImage[resultIndex] )\n" 
    "                        //const float error = _errorImage[resultIndex];\n" 
    "                        #define upstreamDataIndex ( upstreamRow * gInputImageSize + upstreamCol )\n" 
    "                        #define upstreamResult ( _imageImage[upstreamDataIndex] )\n" 
    "                        thiswchange += upstreamResult * error;\n" 
    "    #ifdef BIASED\n" 
    "                        thisbiaschange += error;\n" 
    "    #endif\n" 
    "                    }\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    if( localId < gFilterSizeSquared ) {\n" 
    "        weights[ workgroupId * gFilterSizeSquared + localId ] -= learningRateMultiplier * thiswchange;\n" 
    "    }\n" 
    "#ifdef BIASED\n" 
    "    #define writeBias ( upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin )\n" 
    "    if( writeBias ) {\n" 
    "        biasWeights[outPlane] -= learningRateMultiplier * thisbiaschange;\n" 
    "    }\n" 
    "#endif\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backprop_floats_withscratch_dobias", options, "cl/BackpropWeights2Scratch.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "backpropweights2.cl", "backprop_floats_withscratch_dobias", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcGradInput", options );
}

