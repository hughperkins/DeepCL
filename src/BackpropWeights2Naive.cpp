// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeights2Naive.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeights2Naive::~BackpropWeights2Naive() {
//    cout << "~backpropweights2naive: deleting kernel" << endl;
    delete kernel;
}
VIRTUAL void BackpropWeights2Naive::backpropWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeights2Naive start" );

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

    int globalSize = dim.filtersSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeights2Naive end" );
}
BackpropWeights2Naive::BackpropWeights2Naive( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
    std::string options = dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/backpropweights2.cl", "backprop_floats", 'options' )
    // ]]]
    // generated using cog, from cl/backpropweights2.cl:
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
    "// globalId: [outPlane][inputPlane][filterRow][filterCol]\n" 
    "// per-thread iteration: [n][outputRow][outputCol]\n" 
    "void kernel backprop_floats( const float learningRateMultiplier,\n" 
    "        const int batchSize,\n" 
    "         global const float *gradOutput, global const float *images,\n" 
    "        global float *weights\n" 
    "        #ifdef BIASED\n" 
    "            , global float *biasWeights\n" 
    "        #endif\n" 
    " ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "    if( globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize ) {\n" 
    "        return;\n" 
    "    }\n" 
    "\n" 
    "    int IntraFilterOffset = globalId % gFilterSizeSquared;\n" 
    "    int filterRow = IntraFilterOffset / gFilterSize;\n" 
    "    int filterCol = IntraFilterOffset % gFilterSize;\n" 
    "\n" 
    "    int filter2Id = globalId / gFilterSizeSquared;\n" 
    "    int outPlane = filter2Id / gInputPlanes;\n" 
    "    int upstreamPlane = filter2Id % gInputPlanes;\n" 
    "\n" 
    "    float thiswchange = 0;\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "#ifdef BIASED\n" 
    "    float thisbiaschange = 0;\n" 
    "#endif\n" 
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        for( int outRow = 0; outRow < gOutputImageSize; outRow++ ) {\n" 
    "            int upstreamRow = outRow - gMargin + filterRow;\n" 
    "            for( int outCol = 0; outCol < gOutputImageSize; outCol++ ) {\n" 
    "                int upstreamCol = outCol - gMargin + filterCol;\n" 
    "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputImageSize\n" 
    "                    && upstreamCol < gInputImageSize;\n" 
    "                if( proceed ) {\n" 
    "                    int resultIndex = ( ( n * gNumFilters\n" 
    "                              + outPlane ) * gOutputImageSize\n" 
    "                              + outRow ) * gOutputImageSize\n" 
    "                              + outCol;\n" 
    "                    float error = gradOutput[resultIndex];\n" 
    "                    int upstreamDataIndex = ( ( n * gInputPlanes\n" 
    "                                     + upstreamPlane ) * gInputImageSize\n" 
    "                                     + upstreamRow ) * gInputImageSize\n" 
    "                                     + upstreamCol;\n" 
    "                    float upstreamResult = images[upstreamDataIndex];\n" 
    "                    float thisimagethiswchange = upstreamResult * error;\n" 
    "                    thiswchange += thisimagethiswchange;\n" 
    "    #ifdef BIASED\n" 
    "                    thisbiaschange += error;\n" 
    "    #endif\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "    weights[ globalId ] += - learningRateMultiplier * thiswchange;\n" 
    "#ifdef BIASED\n" 
    "    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n" 
    "    if( writeBias ) {\n" 
    "        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;\n" 
    "    }\n" 
    "#endif\n" 
    "}\n" 
    "\n" 
    "\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backprop_floats", options, "cl/backpropweights2.cl" );
    // [[[end]]]
}

