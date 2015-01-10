#include "BackpropWeightsCpu.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsCpu::BackpropWeightsCpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) :
        BackpropWeights( cl, dim, fn )
            {
}
VIRTUAL BackpropWeightsCpu::~BackpropWeightsCpu() {
}
VIRTUAL void BackpropWeightsCpu::backpropWeights( int batchSize, float learningRate,  CLWrapper *errorsWrapper, CLWrapper *resultsWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    throw std::runtime_error("backpropWeights wrappers not implemented for BackpropWeightsCpu");
}
VIRTUAL void BackpropWeightsCpu::backpropWeights( int batchSize, float learningRate, float *errors,
    float *results, float *images, float *weights, float *biasWeights ) {

    StatefulTimer::instance()->timeCheck(" BackpropWeightsCpu start" );

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    for( int outPlane = 0; outPlane < dim.numFilters; outPlane++ ) {
        for( int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++ ) {
            for( int filterRow = 0; filterRow < dim.filterSize; filterRow++ ) {
                for( int filterCol = 0; filterCol <dim.filterSize; filterCol++ ) {
                    int weightIndex = ( ( outPlane
                        * dim.inputPlanes + upstreamPlane )
                        * dim.filterSize + filterRow )
                        * dim.filterSize + filterCol;
                    float thiswchange = 0;
                    float thisBiasChange = 0;
                    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
                    //       aggregate over:  [outRow][outCol][n]
                    for( int outRow = 0; outRow < dim.outputBoardSize; outRow++ ) {
                        int upstreamRow = outRow - margin + filterRow;
                        if( upstreamRow < 0 || upstreamRow > dim.inputBoardSize - 1 ) {
                            continue;
                        }
                        for( int outCol = 0; outCol < dim.outputBoardSize; outCol++ ) {
                            int upstreamCol = outCol - margin + filterCol;
                            if( upstreamCol < 0 || upstreamCol > dim.inputBoardSize - 1 ) {
                                continue;
                            }
                            for( int n = 0; n < batchSize; n++ ) {
                                int resultIndex = ( ( n
                                    * dim.numFilters + outPlane )
                                    * dim.outputBoardSize + outRow )
                                    * dim.outputBoardSize + outCol;
                                float error = errors[resultIndex];
                                float actualOutput = results[resultIndex];
                                float activationDerivative = fn->calcDerivative( actualOutput );
                                int upstreamResultIndex = ( ( n
                                    * dim.inputPlanes + upstreamPlane )
                                    * dim.inputBoardSize + upstreamRow )
                                    * dim.inputBoardSize + upstreamCol;
                                float upstreamResult = images[ upstreamResultIndex ];
                                float thisimagethiswchange = upstreamResult * activationDerivative *
                                    error;
                                thiswchange += thisimagethiswchange;
                                thisBiasChange += activationDerivative;
//    if( ( filterRow * 3 + filterCol ) == 5 ) {
//        cout << thisimagethiswchange << " " << activationDerivative << " " << upstreamResult << " " << error << endl;
//    }
                            }
                        }
                    }
                    weights[ weightIndex ] += - thiswchange * learningMultiplier;
                    if( dim.biased ) {
                        if( filterRow == 0 && filterCol == 0 && upstreamPlane == 0 ) {
                            biasWeights[ outPlane ] += - learningMultiplier * thisBiasChange;
                        }
                    }
                }
            }
        }
    }
    StatefulTimer::instance()->timeCheck(" BackpropWeightsCpu end" );
}

