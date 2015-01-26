// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "Propagate4.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

Propagate4::Propagate4( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
//    kernel = cl->buildKernel( "propagate.cl", "propagate_3_by_n_outplane", options );
    kernel = cl->buildKernel( "propagate.cl", "propagate_4_by_n_outplane_smallercache", options );

//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}
VIRTUAL Propagate4::~Propagate4() {
    delete kernel;
}
VIRTUAL void Propagate4::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//    cout << "square(dim.outputBoardSize) " << square( dim.outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) );

    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
    cout << "propagate4 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate4::propagate after call propagate");
}

