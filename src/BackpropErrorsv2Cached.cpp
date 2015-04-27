#include "StatefulTimer.h"

#include "BackpropErrorsv2Cached.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropErrorsv2Cached::~BackpropErrorsv2Cached() {
    delete kernel;
//    delete applyActivationDeriv;
}
VIRTUAL void BackpropErrorsv2Cached::backward( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached start" );

//        const int batchSize,
//        global const float *gradOutputGlobal,
//        global const float *filtersGlobal, 
//        global float *gradInput,
//        local float *_errorImage, 
//        local float *_filterImage ) {

    kernel
       ->in( batchSize )
        ->in( gradOutputWrapper )
       ->in( weightsWrapper )
        ->out( gradInputWrapper )
        ->localFloats( square( dim.outputImageSize ) )
        ->localFloats( square( dim.filterSize ) );

    int numWorkgroups = batchSize * dim.inputPlanes;
    int workgroupSize = square( dim.inputImageSize );
    workgroupSize = std::max( 32, workgroupSize ); // no point in wasting cores...
    int globalSize = numWorkgroups * workgroupSize;

//    int globalSize = batchSize * dim.inputCubeSize;
//    int workgroupsize = cl->getMaxWorkgroupSize();
//    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    kernel->run_1d(globalSize, workgroupsize);
    
//    float const*gradInput = (float *)gradInputWrapper->getHostArray();
    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();
//    gradInputWrapper->copyToHost();
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached after first kernel" );
//    for( int i = 0; i < min( 40, batchSize * dim.inputCubeSize ); i++ ) {
//        cout << "efu[" << i << "]=" << gradInput[i] << endl;
//    }

//    applyActivationDeriv->in( batchSize * dim.inputCubeSize )->in( gradInputWrapper )->in( inputDataWrapper );
//    applyActivationDeriv->run_1d(globalSize, workgroupSize);
//    applyActivationDeriv->in( batchSize * dim.inputCubeSize )->inout( gradInputWrapper )->in( inputDataWrapper );
//    applyActivationDeriv->run_1d(globalSize, workgroupSize);
//    cl->finish();
//    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached after applyActivationDeriv" );
//    gradInputWrapper->copyToHost();
//    for( int i = 0; i < min( 40, batchSize * dim.inputCubeSize ); i++ ) {
//        cout << "efu2[" << i << "]=" << gradInput[i] << endl;
//    }
    
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached end" );
}
BackpropErrorsv2Cached::BackpropErrorsv2Cached( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropErrorsv2( cl, dim )
            {
    std::string options = dim.buildOptionsString();
    options += ""; // " -D " + upstreamFn->getDefineName();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/backproperrorsv2cached.cl", "calcGradInputCached", 'options' )
    // # stringify.write_kernel2( "broadcastMultiply", "cl/backproperrorsv2.cl", "broadcast_multiply", 'options' )
    // # stringify.write_kernel2( "applyActivationDeriv", "cl/applyActivationDeriv.cl", "applyActivationDeriv", 'options' )
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // generated using cog, from cl/backproperrorsv2cached.cl:
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
    "// as calcGradInput, but with local cache\n" 
    "// convolve weights with gradOutput to produce gradInput\n" 
    "// workgroupid: [n][inputPlane]\n" 
    "// localid: [upstreamrow][upstreamcol]\n" 
    "// per-thread aggregation: [outPlane][filterRow][filterCol]\n" 
    "// need to store locally:\n" 
    "// - _errorImage. size = outputImageSizeSquared\n" 
    "// - _filterImage. size = filtersizesquared\n" 
    "// note: currently doesnt use bias as input.  thats probably an error?\n" 
    "// inputs: gradOutput :convolve: filters => gradInput\n" 
    "//\n" 
    "// global:\n" 
    "// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4\n" 
    "// weights: [filterId][upstreamplane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4\n" 
    "// per workgroup:\n" 
    "// gradOutput: [outPlane][outRow][outCol] 32 * 19 * 19 * 4 = 46KB\n" 
    "// weights: [filterId][filterRow][filterCol] 32 * 5 * 5 * 4 = 3.2KB\n" 
    "// gradOutputforupstream: [n][upstreamPlane][upstreamRow][upstreamCol]\n" 
    "void kernel calcGradInputCached(\n" 
    "        const int batchSize,\n" 
    "        global const float *gradOutputGlobal,\n" 
    "        global const float *filtersGlobal,\n" 
    "        global float *gradInput,\n" 
    "        local float *_errorImage,\n" 
    "        local float *_filterImage ) {\n" 
    "\n" 
    "    #define globalId get_global_id(0)\n" 
    "    #define localId get_local_id(0)\n" 
    "    #define workgroupId get_group_id(0)\n" 
    "    #define workgroupSize get_local_size(0)\n" 
    "\n" 
    "    const int n = workgroupId / gInputPlanes;\n" 
    "    const int upstreamPlane = workgroupId % gInputPlanes;\n" 
    "\n" 
    "    const int upstreamRow = localId / gInputImageSize;\n" 
    "    const int upstreamCol = localId % gInputImageSize;\n" 
    "\n" 
    "    float sumWeightTimesOutError = 0;\n" 
    "    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        copyLocal( _filterImage, filtersGlobal + ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared, gFilterSizeSquared );\n" 
    "        copyLocal( _errorImage, gradOutputGlobal + ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared, gOutputImageSizeSquared );\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {\n" 
    "            int outRow = upstreamRow + gMargin - filterRow;\n" 
    "            for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {\n" 
    "                int outCol = upstreamCol + gMargin - filterCol;\n" 
    "                if( outCol >= 0 && outCol < gOutputImageSize && outRow >= 0 && outRow < gOutputImageSize ) {\n" 
    "                    float thisWeightTimesError =\n" 
    "                        _errorImage[outRow * gOutputImageSize + outCol] *\n" 
    "                        _filterImage[filterRow * gFilterSize + filterCol];\n" 
    "                    sumWeightTimesOutError += thisWeightTimesError;\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    const int upstreamImageGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared;\n" 
    "    if( localId < gInputImageSizeSquared ) {\n" 
    "        gradInput[upstreamImageGlobalOffset + localId] = sumWeightTimesOutError;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "calcGradInputCached", options, "cl/backproperrorsv2cached.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "backproperrorsv2.cl", "calcGradInput", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcGradInput", options );
}

