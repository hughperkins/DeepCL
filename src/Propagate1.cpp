#include "Propagate1.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

Propagate1::Propagate1( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
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
    kernel = cl->buildKernel( "propagate.cl", "convolve_imagecubes_float2", options );
//    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
}
VIRTUAL Propagate1::~Propagate1() {
    delete kernel;
}
VIRTUAL void Propagate1::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    kernel->in(batchSize)
        ->in( dim.inputPlanes )->in( dim.numFilters )
        ->in( dim.inputBoardSize )->in( dim.filterSize )
       ->in( dim.padZeros ? 1 : 0 );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );

    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = std::min( globalSize, cl->getMaxWorkgroupSize() );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    cout << "propagate1 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate1::propagate after call propagate");
}

