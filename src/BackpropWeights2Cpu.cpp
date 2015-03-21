// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeights2Cpu.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeights2Cpu::BackpropWeights2Cpu( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights2( cl, dim )
            {
}
VIRTUAL BackpropWeights2Cpu::~BackpropWeights2Cpu() {
}
VIRTUAL void BackpropWeights2Cpu::backpropWeights( int batchSize, float learningRate,  CLWrapper *derivLossBySumWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper ) {
    derivLossBySumWrapper->copyToHost();
    imagesWrapper->copyToHost();
    float *biasWeights = 0;
    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
        biasWeights =  (float *)biasWeightsWrapper->getHostArray();
    }
    backpropWeights( batchSize, learningRate, (float *)derivLossBySumWrapper->getHostArray(), (float *)imagesWrapper->getHostArray(),
        (float *)weightsWrapper->getHostArray(), biasWeights );
    weightsWrapper->copyToDevice();
    if( dim.biased ) {
        biasWeightsWrapper->copyToDevice();
    }
}
VIRTUAL void BackpropWeights2Cpu::backpropWeights( int batchSize, float learningRate, float *derivLossBySum,
    float *images, float *weights, float *biasWeights ) {

    StatefulTimer::instance()->timeCheck(" BackpropWeights2Cpu start" );

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    cout << "dim.inputBoardSize=" << dim.inputBoardSize << endl;
    cout << "dim.outputBoardSize=" << dim.outputBoardSize << endl;
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
                    for( int outRow = 0; outRow < dim.outputImageSize; outRow++ ) {
                        int upstreamRow = outRow - margin + filterRow;
                        if( upstreamRow < 0 || upstreamRow > dim.inputImageSize - 1 ) {
                            continue;
                        }
                        for( int outCol = 0; outCol < dim.outputImageSize; outCol++ ) {
                            int upstreamCol = outCol - margin + filterCol;
                            if( upstreamCol < 0 || upstreamCol > dim.inputImageSize - 1 ) {
                                continue;
                            }
                            for( int n = 0; n < batchSize; n++ ) {
                                int resultIndex = ( ( n
                                    * dim.numFilters + outPlane )
                                    * dim.outputImageSize + outRow )
                                    * dim.outputImageSize + outCol;
                                float _derivLossBySum = derivLossBySum[resultIndex];
                                int upstreamResultIndex = ( ( n
                                    * dim.inputPlanes + upstreamPlane )
                                    * dim.inputImageSize + upstreamRow )
                                    * dim.inputImageSize + upstreamCol;
                                float upstreamResult = images[ upstreamResultIndex ];
                                float thisimagethiswchange = _derivLossBySum * upstreamResult;
                                thiswchange += thisimagethiswchange;
                                thisBiasChange += _derivLossBySum; // fairly sure this is right.  Fairly :-P
                                if( outPlane == 0 && filterRow ==  margin && filterCol == margin && upstreamPlane == 0 ) {
                                    cout << "_derivLossBySum " << upstreamRow << "," << upstreamCol << "=" << _derivLossBySum << endl;
                                }
                            }
                        }
                    }
//                    cout << "weight change " << weightIndex << " " << learningMultiplier * thiswchange << endl;
                    weights[ weightIndex ] += - thiswchange * learningMultiplier;
                    if( dim.biased ) {
                        if( filterRow == margin && filterCol == margin && upstreamPlane == 0 ) {
                            biasWeights[ outPlane ] += - learningMultiplier * thisBiasChange;
                        }
                    }
                }
            }
        }
    }
    cout << endl;
    StatefulTimer::instance()->timeCheck(" BackpropWeights2Cpu end" );
}

