// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackpropWeights2ScratchLarge.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeights2ScratchLarge::~BackpropWeights2ScratchLarge() {
    delete kernel;
}
VIRTUAL void BackpropWeights2ScratchLarge::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeights2ScratchLarge start" );

    int workgroupSize = 32 * ( ( square(dim.filterSize) + 32 - 1 ) / 32 ); // quantize to nearest 32
//    int workgroupsize = std::max( 32, square( dim.filterSize ) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupSize * numWorkgroups;
//    globalSize = ( ( globalSize + workgroupSize - 1 ) / workgroupSize ) * workgroupSize;
//    cout << "workgroupsize " << workgroupSize << " numworkgroups " << numWorkgroups << " globalsize " << globalSize << endl;

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
        ->localFloats( outputStripeSize )
        ->localFloats( inputStripeOuterSize );

    kernel->run_1d(globalSize, workgroupSize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeights2ScratchLarge end" );
}
BackpropWeights2ScratchLarge::BackpropWeights2ScratchLarge( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
//    cout << "dim: " << dim << endl;
    std::string options = dim.buildOptionsString();

    int localMemoryRequirementsFullImage = dim.inputBoardSize * dim.inputBoardSize * 4 + dim.outputBoardSize * dim.outputBoardSize * 4;
    int availableLocal = cl->getLocalMemorySize();
    cout << "localmemoryrequirementsfullimage: " << localMemoryRequirementsFullImage << endl;
    cout << "availablelocal: " << availableLocal << endl;
    // make the local memory used about one quarter of what is available? half of what is available?
    // let's try one quarter :-)
    int localWeCanUse = availableLocal / 4;
    numStripes = ( localMemoryRequirementsFullImage + localWeCanUse - 1 ) / localWeCanUse;
//    cout << "numStripes: " << numStripes << endl;
    // make it a power of 2
    numStripes = OpenCLHelper::getNextPower2( numStripes );
//    cout << "numStripes: " << numStripes << endl;

    int inputStripeMarginRows = dim.filterSize - 1;
    int inputStripeInnerNumRows = dim.inputBoardSize / numStripes;
    int inputStripeOuterNumRows = inputStripeInnerNumRows + 2 * inputStripeMarginRows;

    int inputStripeInnerSize = inputStripeInnerNumRows * dim.inputBoardSize;
    inputStripeOuterSize = inputStripeOuterNumRows * dim.inputBoardSize;
    int inputStripeMarginSize = inputStripeMarginRows * dim.inputBoardSize;

    int outputStripeNumRows = ( dim.outputBoardSize + numStripes - 1 ) / numStripes;
    outputStripeSize = outputStripeNumRows * dim.outputBoardSize;

    // [[[cog
    // import cog_optionswriter
    // cog_optionswriter.write_options( ['numStripes','inputStripeMarginRows','inputStripeInnerNumRows',
    //     'inputStripeOuterNumRows', 'inputStripeInnerSize', 'inputStripeOuterSize', 'inputStripeMarginSize',
    //     'outputStripeNumRows', 'outputStripeSize' ] )
    // ]]]
    // generated, using cog:
    options += " -DgNumStripes=" + toString( numStripes );
    options += " -DgInputStripeMarginRows=" + toString( inputStripeMarginRows );
    options += " -DgInputStripeInnerNumRows=" + toString( inputStripeInnerNumRows );
    options += " -DgInputStripeOuterNumRows=" + toString( inputStripeOuterNumRows );
    options += " -DgInputStripeInnerSize=" + toString( inputStripeInnerSize );
    options += " -DgInputStripeOuterSize=" + toString( inputStripeOuterSize );
    options += " -DgInputStripeMarginSize=" + toString( inputStripeMarginSize );
    options += " -DgOutputStripeNumRows=" + toString( outputStripeNumRows );
    options += " -DgOutputStripeSize=" + toString( outputStripeSize );
    // [[[end]]]
    cout << "options: " << options << endl;

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/BackpropWeights2ScratchLarge.cl", "backprop_floats_withscratch_dobias_striped", 'options' )
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
    "// workgroupId: [outputPlane][inputPlane]\n" 
    "// localId: [filterRow][filterCol]\n" 
    "// per-thread iteration: [n][outputRow][outputCol]\n" 
    "// local: errorboard: outputBoardSize * outputBoardSize\n" 
    "//        imageboard: inputBoardSize * inputBoardSize\n" 
    "// specific characteristic: load one stripe of each image at a time,\n" 
    "// so we dont run out of memory\n" 
    "// number of stripes set in: gNumStripes\n" 
    "// note that whilst we can stripe the errors simply,\n" 
    "// we actually need to add a half-filter widthed additional few rows\n" 
    "// onto the images stripe, otherwise we will be missing data\n" 
    "//   we will call the size of the non-overlapping image stripes: gInputStripeInnerSize\n" 
    "//      the outersize, including the two margins is: gInputStripeOuterSize\n" 
    "//      of course, the first and last stripes will be missing a bit off the top/bottom, where the\n" 
    "//      corresponding outer margin would be\n" 
    "void kernel backprop_floats_withscratch_dobias_striped(\n" 
    "        const float learningRateMultiplier, const int batchSize,\n" 
    "         global const float *errors, global const float *images,\n" 
    "        global float *weights,\n" 
    "        #ifdef BIASED\n" 
    "             global float *biasWeights,\n" 
    "        #endif\n" 
    "        local float *_errorStripe, local float *_imageStripe\n" 
    " ) {\n" 
    "    // gHalfFilterSize\n" 
    "    // gInputBoardSize\n" 
    "    //\n" 
    "    // gInputStripeMarginRows => basically equal to gHalfFilterSize\n" 
    "    // gInputStripeInnerNumRows = gInputBoardSize / gNumStripes\n" 
    "    // gInputStripeOuterNumRows = gInputStripeInnerNumRows + 2 * gHalfFilterSize  (note: one row less than\n" 
    "    //                                                         if we just added gFilterSize)\n" 
    "    // gInputStripeInnerSize = gInputStripeInnerNumRows * gInputBoardSize\n" 
    "    // gInputStripeOuterSize = gInputStripeOuterNumRows * gInputBoardSize\n" 
    "    // gInputStripeMarginSize = gInputStripeMarginRows * gInputBoardSize\n" 
    "    //\n" 
    "    // gOutputStripeNumRows\n" 
    "    // gOutputStripeSize\n" 
    "\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "\n" 
    "    const int filterRow = localId / gFilterSize;\n" 
    "    const int filterCol = localId % gFilterSize;\n" 
    "\n" 
    "    const int outPlane = workgroupId / gInputPlanes;\n" 
    "    const int upstreamPlane = workgroupId % gInputPlanes;\n" 
    "\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "    float thiswchange = 0;\n" 
    "#ifdef BIASED\n" 
    "    float thisbiaschange = 0;\n" 
    "#endif\n" 
    "    const int numLoopsForImageStripe = ( gInputStripeOuterSize + workgroupSize - 1 ) / workgroupSize;\n" 
    "    const int numLoopsForErrorStripe = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;\n" 
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        const int imageBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;\n" 
    "        const int imageBoardGlobalOffsetAfter = imageBoardGlobalOffset + gInputBoardSizeSquared;\n" 
    "        const int errorBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;\n" 
    "        const int errorBoardGlobalOffsetAfter = errorBoardGlobalOffset + gOutputBoardSizeSquared;\n" 
    "        for( int stripe = 0; stripe < gNumStripes; stripe++ ) {\n" 
    "            const int imageStripeInnerOffset = imageBoardGlobalOffset + stripe * gInputStripeInnerSize;\n" 
    "            const int imageStripeOuterOffset = imageStripeInnerOffset - gInputStripeMarginSize;\n" 
    "            // need to fetch the board, but it's bigger than us, so will need to loop...\n" 
    "            barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "            for( int i = 0; i < numLoopsForImageStripe; i++ ) {\n" 
    "                int thisOffset = i * workgroupSize + localId;\n" 
    "                int thisGlobalImagesOffset = imageStripeOuterOffset + thisOffset;\n" 
    "                bool process = thisOffset < gInputStripeOuterSize\n" 
    "                    && thisGlobalImagesOffset >= imageBoardGlobalOffset\n" 
    "                    && thisGlobalImagesOffset < imageBoardGlobalOffsetAfter;\n" 
    "                if( process ) {\n" 
    "                    _imageStripe[thisOffset] = images[ thisGlobalImagesOffset ];\n" 
    "                }\n" 
    "            }\n" 
    "            int errorStripeOffset = errorBoardGlobalOffset + stripe * gOutputStripeSize;\n" 
    "            for( int i = 0; i < numLoopsForErrorStripe; i++ ) {\n" 
    "                int thisOffset = i * workgroupSize + localId;\n" 
    "                int globalErrorsOffset = errorStripeOffset + thisOffset;\n" 
    "                bool process = thisOffset < gOutputStripeSize\n" 
    "                    && globalErrorsOffset < errorBoardGlobalOffsetAfter;\n" 
    "                if( process ) {\n" 
    "                    _errorStripe[thisOffset ] = errors[globalErrorsOffset];\n" 
    "                }\n" 
    "            }\n" 
    "            const int stripeOutRowStart = stripe * gOutputStripeNumRows;\n" 
    "            const int stripeOutRowEndExcl = stripeOutRowStart + gOutputStripeNumRows;\n" 
    "            barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "//            if( localId == 13 ) {\n" 
    "//                for( int i = 0; i < 12; i++ ) {\n" 
    "//                    weights[100 + stripe * 12 + i ] = _errorStripe[i * gOutputBoardSize];\n" 
    "//                }\n" 
    "//                for( int i = 0; i < 20; i++ ) {\n" 
    "//                    weights[200 + stripe * 20 + i ] = _imageStripe[i * gInputBoardSize];\n" 
    "//                }\n" 
    "//            }\n" 
    "            if( localId < gFilterSizeSquared ) {\n" 
    "                for( int outRow = stripeOutRowStart; outRow < stripeOutRowEndExcl; outRow++ ) {\n" 
    "                    int upstreamRow = outRow - gMargin + filterRow;\n" 
    "                    for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {\n" 
    "                        int upstreamCol = outCol - gMargin + filterCol;\n" 
    "                        bool proceed =\n" 
    "                            upstreamRow >= 0 && upstreamCol >= 0\n" 
    "                            && upstreamRow < gInputBoardSize && upstreamCol < gInputBoardSize\n" 
    "                            && outRow < gOutputBoardSize;\n" 
    "                        if( proceed ) {\n" 
    "                            int resultIndex = outRow * gOutputBoardSize + outCol;\n" 
    "                            float error = _errorStripe[resultIndex - stripe * gOutputStripeSize];\n" 
    "                            int upstreamDataIndex = upstreamRow * gInputBoardSize + upstreamCol;\n" 
    "                            float upstreamResult = _imageStripe[upstreamDataIndex +  gInputStripeMarginSize\n" 
    "                                        - stripe * gInputStripeInnerSize ];\n" 
    "                            thiswchange += upstreamResult * error;\n" 
    "        #ifdef BIASED\n" 
    "                            thisbiaschange += error;\n" 
    "        #endif\n" 
    "                        }\n" 
    "                    }\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    if( localId < gFilterSizeSquared ) {\n" 
    "        weights[ workgroupId * gFilterSizeSquared + localId ] += - learningRateMultiplier * thiswchange;\n" 
    "//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;\n" 
    "    }\n" 
    "#ifdef BIASED\n" 
    "    bool writeBias = upstreamPlane == 0 && localId == 0;\n" 
    "    if( writeBias ) {\n" 
    "        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;\n" 
    "    }\n" 
    "#endif\n" 
    "    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]\n" 
    "    //       aggregate over:  [outRow][outCol][n]\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backprop_floats_withscratch_dobias_striped", options, "cl/BackpropWeights2ScratchLarge.cl" );
    // [[[end]]]
}

