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

BackpropWeights2ScratchLarge::BackpropWeights2ScratchLarge( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();

    //  gInputStripeMarginRows => basically equal to gHalfFilterSize
    //  gInputStripeInnerNumRows = gInputBoardSize / gNumStripes
    //  gInputStripeOuterNumRows = gInputStripeInnerNumRows + 2 * gHalfFilterSize  (note: one row less than
    //                                                         if we just added gFilterSize)
    //  gInputStripeInnerSize = gInputStripeInnerNumRows * gInputBoardSize
    //  gInputStripeOuterSize = gInputStripeOuterNumRows * gInputBoardSize
    //  gInputStripeMarginSize = gInputStripeMarginRows * gInputBoardSize
    //
    //  gOutputStripeNumRows
    //  gOutputStripeSize

    int localMemoryRequirementsFullImage = dim.inputBoardSize * dim.inputBoardSize * 4 + dim.outputBoardSize * dim.outputBoardSize * 4;
    int availableLocal = cl->getLocalMemorySize();
    cout << "localmemoryrequirementsfullimage: " << localMemoryRequirementsFullImage << endl;
    cout << "availablelocal: " << availableLocal << endl;
    // make the local memory used about one quarter of what is available? half of what is available?
    // let's try one quarter :-)
    int localWeCanUse = availableLocal / 4;
    numStripes = ( localMemoryRequirementsFullImage + localWeCanUse - 1 ) / localWeCanUse;
    cout << "numStripes: " << numStripes << endl;
    // make it a power of 2
    numStripes = OpenCLHelper::getNextPower2( numStripes );
    cout << "numStripes: " << numStripes << endl;

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

    kernel = cl->buildKernel( "BackpropWeights2ScratchLarge.cl", "backprop_floats_withscratch_dobias_striped", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
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

