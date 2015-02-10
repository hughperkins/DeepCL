#include "StatefulTimer.h"

#include "BackpropErrorsv2Cached.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropErrorsv2Cached::~BackpropErrorsv2Cached() {
    delete kernel;
    delete applyActivationDeriv;
}
VIRTUAL void BackpropErrorsv2Cached::backpropErrors( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *errorsWrapper, CLWrapper *weightsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached start" );

//        const int batchSize,
//        global const float *errorsGlobal,
//        global const float *filtersGlobal, 
//        global float *errorsForUpstream,
//        local float *_errorBoard, 
//        local float *_filterBoard ) {

    kernel
       ->in( batchSize )
        ->in( errorsWrapper )
       ->in( weightsWrapper )
        ->out( errorsForUpstreamWrapper )
        ->localFloats( square( dim.outputBoardSize ) )
        ->localFloats( square( dim.filterSize ) );

    int numWorkgroups = batchSize * dim.inputPlanes;
    int workgroupSize = square( dim.inputBoardSize );
    workgroupSize = std::max( 32, workgroupSize ); // no point in wasting cores...
    int globalSize = numWorkgroups * workgroupSize;

//    int globalSize = batchSize * dim.inputCubeSize;
//    int workgroupsize = cl->getMaxWorkgroupSize();
//    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    kernel->run_1d(globalSize, workgroupsize);
    
    float const*errorsForUpstream = (float *)errorsForUpstreamWrapper->getHostArray();
    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();
//    errorsForUpstreamWrapper->copyToHost();
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached after first kernel" );
//    for( int i = 0; i < min( 40, batchSize * dim.inputCubeSize ); i++ ) {
//        cout << "efu[" << i << "]=" << errorsForUpstream[i] << endl;
//    }

//    applyActivationDeriv->in( batchSize * dim.inputCubeSize )->in( errorsForUpstreamWrapper )->in( inputDataWrapper );
//    applyActivationDeriv->run_1d(globalSize, workgroupSize);
    applyActivationDeriv->in( batchSize * dim.inputCubeSize )->inout( errorsForUpstreamWrapper )->in( inputDataWrapper );
    applyActivationDeriv->run_1d(globalSize, workgroupSize);
    cl->finish();
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached after applyActivationDeriv" );
//    errorsForUpstreamWrapper->copyToHost();
//    for( int i = 0; i < min( 40, batchSize * dim.inputCubeSize ); i++ ) {
//        cout << "efu2[" << i << "]=" << errorsForUpstream[i] << endl;
//    }
    
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cached end" );
}
BackpropErrorsv2Cached::BackpropErrorsv2Cached( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn ) :
        BackpropErrorsv2( cl, dim, upstreamFn )
            {
    std::string options = dim.buildOptionsString();
    options += " -D " + upstreamFn->getDefineName();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/backproperrorsv2cached.cl", "calcErrorsForUpstreamCached", 'options' )
    // # stringify.write_kernel2( "broadcastMultiply", "cl/backproperrorsv2.cl", "broadcast_multiply", 'options' )
    // stringify.write_kernel2( "applyActivationDeriv", "cl/applyActivationDeriv.cl", "applyActivationDeriv", 'options' )
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // generated using cog:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// as calcErrorsForUpstream, but with local cache\n" 
    "// convolve weights with errors to produce errorsForUpstream\n" 
    "// workgroupid: [n][inputPlane]\n" 
    "// localid: [upstreamrow][upstreamcol]\n" 
    "// per-thread aggregation: [outPlane][filterRow][filterCol]\n" 
    "// need to store locally:\n" 
    "// - _errorBoard. size = outputBoardSizeSquared\n" 
    "// - _filterBoard. size = filtersizesquared\n" 
    "// note: currently doesnt use bias as input.  thats probably an error?\n" 
    "// inputs: errors :convolve: filters => errorsForUpstream\n" 
    "//\n" 
    "// global:\n" 
    "// errors: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4\n" 
    "// weights: [filterId][upstreamplane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4\n" 
    "// per workgroup:\n" 
    "// errors: [outPlane][outRow][outCol] 32 * 19 * 19 * 4 = 46KB\n" 
    "// weights: [filterId][filterRow][filterCol] 32 * 5 * 5 * 4 = 3.2KB\n" 
    "// errorsforupstream: [n][upstreamPlane][upstreamRow][upstreamCol]\n" 
    "void kernel calcErrorsForUpstreamCached(\n" 
    "        const int batchSize,\n" 
    "        global const float *errorsGlobal,\n" 
    "        global const float *filtersGlobal,\n" 
    "        global float *errorsForUpstream,\n" 
    "        local float *_errorBoard,\n" 
    "        local float *_filterBoard ) {\n" 
    "\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "\n" 
    "    const int n = workgroupId / gInputPlanes;\n" 
    "    const int upstreamPlane = workgroupId % gInputPlanes;\n" 
    "\n" 
    "    const int upstreamRow = localId / gInputBoardSize;\n" 
    "    const int upstreamCol = localId % gInputBoardSize;\n" 
    "\n" 
    "    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutputBoardSize - 1) );\n" 
    "    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );\n" 
    "    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutputBoardSize -1) );\n" 
    "    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );\n" 
    "\n" 
    "    const int filterPixelCopiesPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "    const int errorPixelCopiesPerThread = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "    const int pixelCopiesPerThread = max( filterPixelCopiesPerThread, errorPixelCopiesPerThread );\n" 
    "\n" 
    "    float sumWeightTimesOutError = 0;\n" 
    "    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {\n" 
    "        const int filterBoardGlobalOffset =( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared;\n" 
    "        const int errorBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        for( int i = 0; i < pixelCopiesPerThread; i++ ) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if( thisOffset < gFilterSizeSquared ) {\n" 
    "                _filterBoard[ thisOffset ] = filtersGlobal[ filterBoardGlobalOffset + thisOffset ];\n" 
    "            }\n" 
    "            if( thisOffset < gOutputBoardSizeSquared ) {\n" 
    "                _errorBoard[ thisOffset ] = errorsGlobal[ errorBoardGlobalOffset + thisOffset ];\n" 
    "            }\n" 
    "        }\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "//        if( globalId == 0 ) {\n" 
    "//            for( int i = 0; i < gFilterSizeSquared; i++ ) {\n" 
    "//                errorsForUpstream[ (outPlane+1)*100 + i ] = _filterBoard[i];\n" 
    "//            }\n" 
    "//        }\n" 
    "        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {\n" 
    "            int outRow = upstreamRow + gMargin - filterRow;\n" 
    "            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {\n" 
    "                int outCol = upstreamCol + gMargin - filterCol;\n" 
    "                int resultIndex = outRow * gOutputBoardSize + outCol;\n" 
    "                float thisError = _errorBoard[resultIndex];\n" 
    "                int thisWeightIndex = filterRow * gFilterSize + filterCol;\n" 
    "                float thisWeight = _filterBoard[thisWeightIndex];\n" 
    "                float thisWeightTimesError = thisWeight * thisError;\n" 
    "                sumWeightTimesOutError += thisWeightTimesError;\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    const int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;\n" 
    "    if( localId < gInputBoardSizeSquared ) {\n" 
    "        errorsForUpstream[upstreamBoardGlobalOffset + localId] = sumWeightTimesOutError;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstreamCached", options, "cl/backproperrorsv2cached.cl" );
    // generated using cog:
    const char * applyActivationDerivSource =  
    "// Copyright Hugh Perkins 201, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH ]\n" 
    "\n" 
    "#ifdef TANH\n" 
    "    #define ACTIVATION_DERIV(output) (1 - output * output)\n" 
    "#elif defined SCALEDTANH\n" 
    "    #define ACTIVATION_DERIV(output) ( 0.66667f * ( 1.7159f - 1 / 1.7159f * output * output ) )\n" 
    "#elif defined SIGMOID\n" 
    "    #define ACTIVATION_DERIV(output) (output * ( 1 - output ) )\n" 
    "#elif defined RELU\n" 
    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)\n" 
    "#elif defined LINEAR\n" 
    "    #define ACTIVATION_DERIV(output) (1.0f)\n" 
    "#endif\n" 
    "\n" 
    "//#ifdef ACTIVATION_DERIV\n" 
    "//void kernel applyActivationDeriv(\n" 
    "//        const int N,\n" 
    "//        global float *inout ) {\n" 
    "//    int globalId = get_global_id(0);\n" 
    "//    inout[globalId] = ACTIVATION_DERIV( inout[globalId] );\n" 
    "//}\n" 
    "//#endif\n" 
    "\n" 
    "#ifdef ACTIVATION_DERIV\n" 
    "void kernel applyActivationDeriv(\n" 
    "        const int N,\n" 
    "        global float *target, global const float *source ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "    if( globalId < N ) {\n" 
    "        target[globalId] *= ACTIVATION_DERIV( source[globalId] );\n" 
    "    }\n" 
    "  //  target[globalId] *= source[globalId];\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    applyActivationDeriv = cl->buildKernelFromString( applyActivationDerivSource, "applyActivationDeriv", options, "cl/applyActivationDeriv.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "backproperrorsv2.cl", "calcErrorsForUpstream", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}

