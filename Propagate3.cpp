#include "Propagate3.h"
#include "stringhelper.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

Propagate3::Propagate3( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = "-D " + fn->getDefineName();
    if( dim.biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(dim.inputBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(square(dim.inputBoardSize));
    options += " -D gFilterSize=" + toString(dim.filterSize);
    options += " -D gFilterSizeSquared=" + toString(square(dim.filterSize));
    options += " -D gOutBoardSize=" + toString(dim.outputBoardSize);
    options += " -D gOutBoardSizeSquared=" + toString(square(dim.outputBoardSize));
    options += " -D gPadZeros=" + toString(dim.padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(dim.numFilters);
    options += " -D gMargin=" + toString(dim.padZeros ? dim.filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( dim.filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(dim.inputPlanes);
    kernel = cl->buildKernel( "../ClConvolve.cl", "convolve_imagecubes_float4", options );
//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}
VIRTUAL Propagate3::~Propagate3() {
    delete kernel;
}
VIRTUAL void Propagate3::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//        cout << "square(outputBoardSize) " << square( outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.numFilters );

    int workgroupsize = square( dim.outputBoardSize );
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
    cout << "numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
}

