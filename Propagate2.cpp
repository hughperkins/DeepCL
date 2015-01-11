#include "Propagate2.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

Propagate2::Propagate2( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {
    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
    kernel = cl->buildKernel( "propagate.cl", "propagate_2_by_outplane", options );
}
VIRTUAL Propagate2::~Propagate2() {
    delete kernel;
}
// only works for small filters
// condition: square( dim.filterSize ) * dim.inputPlanes * 4 < 5000 (about 5KB)
VIRTUAL void Propagate2::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//        cout << "square(outputBoardSize) " << square( outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );
    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    cout << "propagate2 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate2::propagate after call propagate");
}

