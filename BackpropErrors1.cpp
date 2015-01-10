#include "BackpropErrors1.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropErrors1::BackpropErrors1( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropErrors( cl, dim )
            {
    // [[[cog
    // import stringify
    // # stringify.write_kernel( "kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
    std::string options = "";
    if( dim.biased ) {
         options += " -D BIASED";
    }
    kernel = cl->buildKernel( "backproperrors.cl", "calcErrorsForUpstream", options );
//    kernel = cl->buildKernelFromString( kernelSource, "calcErrorsForUpstream", options );
}
VIRTUAL BackpropErrors1::~BackpropErrors1() {
    delete kernel;
}
VIRTUAL void BackpropErrors1::backpropErrors( int batchSize, 
        CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {

    StatefulTimer::instance()->timeCheck("BackpropErrors1 start" );
    kernel
        ->in( dim.inputPlanes )->in( dim.inputBoardSize )->in( dim.filterSize )
        ->in( dim.numFilters )->in( dim.outputBoardSize )
        ->in( dim.padZeros ? 1 : 0 )
        ->in( weightsWrapper )
        ->in( errorsWrapper )
        ->out( errorsForUpstreamWrapper );
    int globalSize = batchSize * dim.inputCubeSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    std::cout << "calcerrorsforupstreamgpu workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropErrors1 end" );
}

