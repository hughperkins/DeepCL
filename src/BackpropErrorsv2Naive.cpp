#include "StatefulTimer.h"

#include "BackpropErrorsv2Naive.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropErrorsv2Naive::~BackpropErrorsv2Naive() {
    delete kernel;
//    delete broadcastMultiply;
    delete applyActivationDeriv;
}
VIRTUAL void BackpropErrorsv2Naive::backpropErrors( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive start" );

    kernel
       ->in( batchSize )
        ->in( gradOutputWrapper )
       ->in( weightsWrapper )
        ->out( gradInputWrapper );

    int globalSize = batchSize * dim.inputCubeSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive after first kernel" );

    applyActivationDeriv->in( batchSize * dim.inputCubeSize )->in( gradInputWrapper )->in( inputDataWrapper );
    applyActivationDeriv->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive after applyActivationDeriv" );
    
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Naive end" );
}
BackpropErrorsv2Naive::BackpropErrorsv2Naive( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn ) :
        BackpropErrorsv2( cl, dim, upstreamFn )
            {
    std::string options = dim.buildOptionsString();
    options += " -D " + upstreamFn->getDefineName();
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/backproperrorsv2.cl", "calcGradInput", 'options' )
    // # stringify.write_kernel2( "broadcastMultiply", "cl/backproperrorsv2.cl", "broadcast_multiply", 'options' )
    // stringify.write_kernel2( "applyActivationDeriv", "cl/applyActivationDeriv.cl", "applyActivationDeriv", 'options' )
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // generated using cog, from cl/backproperrorsv2.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "//  - none\n" 
    "\n" 
    "// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]\n" 
    "// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB\n" 
    "// errors: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB\n" 
    "// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB\n" 
    "void kernel calcGradInput(\n" 
    "        const int batchSize,\n" 
    "        global const float *errors, global float *weights, global float *gradInput ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int upstreamImage2dId = globalId / gInputImageSizeSquared;\n" 
    "\n" 
    "    const int intraImageOffset = globalId % gInputImageSizeSquared;\n" 
    "    const int upstreamRow = intraImageOffset / gInputImageSize;\n" 
    "    const int upstreamCol = intraImageOffset % gInputImageSize;\n" 
    "\n" 
    "    const int upstreamPlane = upstreamImage2dId % gInputPlanes;\n" 
    "    const int n = upstreamImage2dId / gInputPlanes;\n" 
    "\n" 
    "    if( n >= batchSize ) {\n" 
    "        return;\n" 
    "    }\n" 
    "\n" 
    "    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutputImageSize - 1) );\n" 
    "    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );\n" 
    "    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutputImageSize -1) );\n" 
    "    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );\n" 
    "\n" 
    "    float sumWeightTimesOutError = 0;\n" 
    "//    int inputDataIndex = globalId;\n" 
    "//    float inputDataValue = inputData[inputDataIndex];\n" 
    "//    float inputDeriv = ACTIVATION_DERIV( inputDataValue );\n" 
    "    // aggregate over [outPlane][outRow][outCol]\n" 
    "    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {\n" 
    "        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {\n" 
    "            int outRow = upstreamRow + gMargin - filterRow;\n" 
    "            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {\n" 
    "                int outCol = upstreamCol + gMargin - filterCol;\n" 
    "                int resultIndex = ( ( n * gNumFilters\n" 
    "                          + outPlane ) * gOutputImageSize\n" 
    "                          + outRow ) * gOutputImageSize\n" 
    "                          + outCol;\n" 
    "                float thisError = errors[resultIndex];\n" 
    "                int thisWeightIndex = ( ( outPlane * gInputPlanes\n" 
    "                                    + upstreamPlane ) * gFilterSize\n" 
    "                                    + filterRow ) * gFilterSize\n" 
    "                                    + filterCol;\n" 
    "                float thisWeight = weights[thisWeightIndex];\n" 
    "                float thisWeightTimesError = thisWeight * thisError;\n" 
    "                sumWeightTimesOutError += thisWeightTimesError;\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    gradInput[globalId] = sumWeightTimesOutError;\n" 
    "    //gradInput[globalId] = sumWeightTimesOutError * inputDeriv;\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "calcGradInput", options, "cl/backproperrorsv2.cl" );
    // generated using cog, from cl/applyActivationDeriv.cl:
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
    "#ifdef ACTIVATION_DERIV\n" 
    "void kernel backpropErrors(\n" 
    "        const int N,\n" 
    "        global const float *inputs,\n" 
    "        global const float *errors,\n" 
    "        global float *gradInput ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "    if( globalId < N ) {\n" 
    "        gradInput[globalId] = ACTIVATION_DERIV( inputs[globalId] ) * errors[globalId];\n" 
    "            // probably not ideal to have the output and input separate?\n" 
    "    }\n" 
    "  //  target[globalId] *= source[globalId];\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    applyActivationDeriv = cl->buildKernelFromString( applyActivationDerivSource, "applyActivationDeriv", options, "cl/applyActivationDeriv.cl" );
    // [[[end]]]
//    kernel = cl->buildKernel( "backproperrorsv2.cl", "calcGradInput", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcGradInput", options );
}

