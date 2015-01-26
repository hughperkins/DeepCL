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

BackpropWeights2Scratch::BackpropWeights2Scratch( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = dim.buildOptionsString();
    kernel = cl->buildKernel( "backpropweights2.cl", "backprop_floats_withscratch_dobias", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropWeights2Scratch::~BackpropWeights2Scratch() {
    delete kernel;
}
VIRTUAL void BackpropWeights2Scratch::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    StatefulTimer::instance()->timeCheck("BackpropWeights2Scratch start" );

    int workgroupsize = std::max( 32, square( dim.filterSize ) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

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

