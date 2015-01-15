#include "BackpropErrorsCpu.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropErrorsCpu::BackpropErrorsCpu( OpenCLHelper *cl, LayerDimensions dim ) :
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
}
VIRTUAL BackpropErrorsCpu::~BackpropErrorsCpu() {
}
VIRTUAL float *BackpropErrorsCpu::backpropErrors( int batchSize, float *results, float *weights, float *biasWeights,
    float *errors ) {
    float *errorsForUpstream = new float[ batchSize * dim.inputCubeSize ];

//        Timer timer;
    StatefulTimer::instance()->timeCheck("BackpropErrorsCpu start" );
    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    // handle lower layer...
    // errors for upstream look like [n][inPlane][inRow][inCol]
    // need to aggregate over: [outPlane][outRow][outCol] (?)
    // need to backprop errors along each possible weight
    // each upstream feeds to:
    //    - each of our filters (so numPlanes filters)
    //    - each of our outpoint points (so boardSize * boardSize)
    // for our own backprop, we updated weights for:
    //      [outPlane][inPlane][filterRow][filtercol]
    //    aggregating over: [n][outRow][outCol]
    // errors are provider per [n][inPlane][inRow][inCol]
    for( int n = 0; n < batchSize; n++ ) {
        for( int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++ ) {
            for( int upstreamRow = 0; upstreamRow < dim.inputBoardSize; upstreamRow++ ) {
                int minFilterRow = std::max( 0, upstreamRow + margin - (dim.outputBoardSize - 1) );
                int maxFilterRow = std::min( dim.filterSize - 1, upstreamRow + margin );
                for( int upstreamCol = 0; upstreamCol < dim.inputBoardSize; upstreamCol++ ) {
                    float sumWeightTimesOutError = 0;
                    // aggregate over [outPlane][outRow][outCol]
                    int minFilterCol = std::max( 0, upstreamCol + margin - (dim.outputBoardSize -1) );
                    int maxFilterCol = std::min( dim.filterSize - 1, upstreamCol + margin );
                    for( int outPlane = 0; outPlane < dim.numFilters; outPlane++ ) {
                        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
                            int outRow = upstreamRow + margin - filterRow;
                            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                                int outCol = upstreamCol + margin - filterCol;
                                int resultIndex = ( ( n 
                                    * dim.numFilters + outPlane )
                                    * dim.outputBoardSize + outRow )
                                    * dim.outputBoardSize + outCol;
                                float thisError = errors[resultIndex];
                                int thisWeightIndex = ( ( outPlane 
                                    * dim.inputPlanes + upstreamPlane )
                                    * dim.filterSize + filterRow )
                                    * dim.filterSize + filterCol;
                                float thisWeight = weights[thisWeightIndex];
                                float thisWeightTimesError = thisWeight * thisError;
                                sumWeightTimesOutError += thisWeightTimesError;
                            }
                        }
                    }
                    int upstreamResultIndex = ( ( n
                        * dim.inputPlanes + upstreamPlane )
                        * dim.inputBoardSize + upstreamRow )
                        * dim.inputBoardSize + upstreamCol;
                    errorsForUpstream[upstreamResultIndex] = sumWeightTimesOutError;
                }
            }
        }
    }
//        timer.timeCheck("calced errors for upstream");   
    StatefulTimer::instance()->timeCheck("BackpropErrorsCpu end" );

    return errorsForUpstream;
}
VIRTUAL void BackpropErrorsCpu::backpropErrors( int batchSize, 
        CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {
    throw std::runtime_error( "backpropErrors wrappers not implemented for BackpropErrorsCpu");
}

