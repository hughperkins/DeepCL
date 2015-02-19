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
VIRTUAL void BackpropWeights2Scratch::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeights2Scratch start" );

    int workgroupsize = std::max( 32, square( dim.filterSize ) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

    int localMemRequiredKB = ( square( dim.outputBoardSize ) * 4 + square( dim.inputBoardSize ) * 4 ) / 1024 + 1;
    if( localMemRequiredKB >= cl->getLocalMemorySizeKB() ) {
        throw runtime_error( "local memory too small to use this kernel on this device.  Need: " + 
            toString( localMemRequiredKB ) + "KB, but only have: " + 
            toString( cl->getLocalMemorySizeKB() ) + "KB local memory" );
    }

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
       ->in( errorsWrapper )
        ->in( imagesWrapper )
       ->inout( weightsWrapper );
    if( dim.biased ) {
        kernel->inout( biasWeightsWrapper );
    }
    kernel
        ->localFloats( square( dim.outputBoardSize ) )
        ->localFloats( square( dim.inputBoardSize ) );

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
    // generated using cog:
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
    "\n" 
    "// workgroupId: [outputPlane][inputPlane]\n" 
    "// localId: [filterRow][filterCol]\n" 
    "// per-thread iteration: [n][outputRow][outputCol]\n" 
    "// local: errorboard: outputBoardSize * outputBoardSize\n" 
    "//        imageboard: inputBoardSize * inputBoardSize\n" 
    "void kernel backprop_floats_withscratch_dobias(\n" 
    "        const float learningRateMultiplier, const int batchSize,\n" 
    "         global const float *errors, global const float *images,\n" 
    "        global float *weights,\n" 
    "        #ifdef BIASED\n" 
    "             global float *biasWeights,\n" 
    "        #endif\n" 
    "        local float *_errorBoard, local float *_imageBoard\n" 
    " ) {\n" 
    "    #define globalId ( get_global_id(0) )\n" 
    "    #define localId ( get_local_id(0)  )\n" 
    "    #define workgroupId ( get_group_id(0) )\n" 
    "    #define workgroupSize ( get_local_size(0) )\n" 
    "\n" 
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
    "        copyLocal( _imageBoard, images + ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared, gInputBoardSizeSquared );\n" 
    "        copyLocal(_errorBoard, errors + ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared, gOutputBoardSizeSquared );\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        if( localId < gFilterSizeSquared ) {\n" 
    "            for( int outRow = 0; outRow < gOutputBoardSize; outRow++ ) {\n" 
    "                int upstreamRow = outRow - gMargin + filterRow;\n" 
    "                for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {\n" 
    "                    const int upstreamCol = outCol - gMargin + filterCol;\n" 
    "                    #define proceed ( upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputBoardSize && upstreamCol < gInputBoardSize )\n" 
    "                    if( proceed ) {\n" 
    "                        // these defines reduce register pressure, compared to const\n" 
    "                        // giving a 40% speedup on nvidia :-)\n" 
    "                        #define resultIndex ( outRow * gOutputBoardSize + outCol )\n" 
    "                        #define error ( _errorBoard[resultIndex] )\n" 
    "                        //const float error = _errorBoard[resultIndex];\n" 
    "                        #define upstreamDataIndex ( upstreamRow * gInputBoardSize + upstreamCol )\n" 
    "                        #define upstreamResult ( _imageBoard[upstreamDataIndex] )\n" 
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
    "    #define writeBias ( upstreamPlane == 0 && localId == 0 )\n" 
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
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}

