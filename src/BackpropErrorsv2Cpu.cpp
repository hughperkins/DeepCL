// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropErrorsv2Cpu.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropErrorsv2Cpu::BackpropErrorsv2Cpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn ) :
        BackpropErrorsv2( cl, dim, upstreamFn )
            {
}
VIRTUAL BackpropErrorsv2Cpu::~BackpropErrorsv2Cpu() {
}
VIRTUAL float *BackpropErrorsv2Cpu::backpropErrors( int batchSize, float *inputData,
    float *errors, float *weights ) {
    float *errorsForUpstream = new float[ batchSize * dim.inputCubeSize ];

//        Timer timer;
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cpu start" );
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
//    ActivationFunction *upstreamActivation = 
    for( int n = 0; n < batchSize; n++ ) {
        for( int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++ ) {
            for( int upstreamRow = 0; upstreamRow < dim.inputBoardSize; upstreamRow++ ) {
                int minFilterRow = std::max( 0, upstreamRow + margin - (dim.outputBoardSize - 1) );
                int maxFilterRow = std::min( dim.filterSize - 1, upstreamRow + margin );
                for( int upstreamCol = 0; upstreamCol < dim.inputBoardSize; upstreamCol++ ) {
                    float sumWeightTimesOutError = 0;
                    int inputDataIndex = ( ( n
                        * dim.inputPlanes + upstreamPlane )
                        * dim.inputBoardSize + upstreamRow )
                        * dim.inputBoardSize + upstreamCol;
                    float inputValue = inputData[inputDataIndex];
                    float activationDerivativeUpstream = upstreamFn->calcDerivative(inputValue);
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
                                sumWeightTimesOutError += thisWeight * thisError;
                            }
                        }
                    }
                    int upstreamResultIndex = ( ( n
                        * dim.inputPlanes + upstreamPlane )
                        * dim.inputBoardSize + upstreamRow )
                        * dim.inputBoardSize + upstreamCol;
                    errorsForUpstream[upstreamResultIndex] = sumWeightTimesOutError * activationDerivativeUpstream;
                }
            }
        }
    }
//        timer.timeCheck("calced errors for upstream");   
    StatefulTimer::instance()->timeCheck("BackpropErrorsv2Cpu end" );

    return errorsForUpstream;
}
VIRTUAL void BackpropErrorsv2Cpu::backpropErrors( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *errorsWrapper, CLWrapper *weightsWrapper,
        CLWrapper *errorsForUpstreamWrapper ) {

    inputDataWrapper->copyToHost();
    errorsWrapper->copyToHost();
    weightsWrapper->copyToHost();
//    float *biasWeights = 0;
//    if( dim.biased ) {
//        biasWeightsWrapper->copyToHost();
//        biasWeights =  (float *)biasWeightsWrapper->getHostArray();
//    }
    float *errorsForUpstream = backpropErrors( batchSize, (float *)inputDataWrapper->getHostArray(),
         (float *)errorsWrapper->getHostArray(), (float *)weightsWrapper->getHostArray() );
    float *errorsForUpstreamHostArray = (float*)errorsForUpstreamWrapper->getHostArray();
    const int errorsForUpstreamWrapperSize = errorsForUpstreamWrapper->size();
    for( int i = 0; i < errorsForUpstreamWrapperSize; i++ ) {
        errorsForUpstreamHostArray[i] = errorsForUpstream[i];
    }
    errorsForUpstreamWrapper->copyToDevice();
    delete[] errorsForUpstream;
}

