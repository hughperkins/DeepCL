// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsCpu.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsCpu::BackpropWeightsCpu( OpenCLHelper *cl, LayerDimensions dim ) :
        BackpropWeights( cl, dim )
            {
}
VIRTUAL BackpropWeightsCpu::~BackpropWeightsCpu() {
}
VIRTUAL void BackpropWeightsCpu::calcGradWeights( int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWeightsWrapper ) {
    gradOutputWrapper->copyToHost();
    imagesWrapper->copyToHost();
    float *gradBiasWeights = 0;
    if( dim.biased ) {
        gradBiasWeightsWrapper->copyToHost();
        gradBiasWeights =  (float *)gradBiasWeightsWrapper->getHostArray();
    }
    backpropWeights( batchSize, learningRate, (float *)gradOutputWrapper->getHostArray(), (float *)imagesWrapper->getHostArray(),
        (float *)gradWeightsWrapper->getHostArray(), gradBiasWeights );
    gradWeightsWrapper->copyToDevice();
    if( dim.biased ) {
        gradBiasWeightsWrapper->copyToDevice();
    }
}
VIRTUAL void BackpropWeightsCpu::backpropWeights( int batchSize, float learningRate, float *gradOutput,
    float *input, float *gradWeights, float *gradBiasWeights ) {

    StatefulTimer::instance()->timeCheck(" BackpropWeightsCpu start" );

    const float learningMultiplier = learningRateToMultiplier( batchSize, learningRate );

    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    for( int outPlane = 0; outPlane < dim.numFilters; outPlane++ ) {
        for( int inputPlane = 0; inputPlane < dim.inputPlanes; inputPlane++ ) {
            for( int filterRow = 0; filterRow < dim.filterSize; filterRow++ ) {
                for( int filterCol = 0; filterCol <dim.filterSize; filterCol++ ) {
                    int weightIndex = ( ( outPlane
                        * dim.inputPlanes + inputPlane )
                        * dim.filterSize + filterRow )
                        * dim.filterSize + filterCol;
                    float thiswchange = 0;
                    float thisBiasChange = 0;
                    // gradWeights:     [outPlane][inputPlane][filterRow][filterCol]
                    //       aggregate over:  [outRow][outCol][n]
                    for( int outRow = 0; outRow < dim.outputImageSize; outRow++ ) {
                        int inputRow = outRow - margin + filterRow;
                        if( inputRow < 0 || inputRow > dim.inputImageSize - 1 ) {
                            continue;
                        }
                        for( int outCol = 0; outCol < dim.outputImageSize; outCol++ ) {
                            int inputCol = outCol - margin + filterCol;
                            if( inputCol < 0 || inputCol > dim.inputImageSize - 1 ) {
                                continue;
                            }
                            for( int n = 0; n < batchSize; n++ ) {
                                int resultIndex = ( ( n
                                    * dim.numFilters + outPlane )
                                    * dim.outputImageSize + outRow )
                                    * dim.outputImageSize + outCol;
                                float _gradOutput = gradOutput[resultIndex];
                                int inputIndex = ( ( n
                                    * dim.inputPlanes + inputPlane )
                                    * dim.inputImageSize + inputRow )
                                    * dim.inputImageSize + inputCol;
                                float inputValue = input[ inputIndex ];
                                float thisimagethiswchange = _gradOutput * inputValue;
                                thiswchange += thisimagethiswchange;
                                thisBiasChange += _gradOutput; // fairly sure this is right.  Fairly :-P
                            }
                        }
                    }
//                    cout << "weight change " << weightIndex << " " << learningMultiplier * thiswchange << endl;
                    gradWeights[ weightIndex ] = thiswchange * learningMultiplier;
                    if( dim.biased ) {
                        if( filterRow == margin && filterCol == margin && inputPlane == 0 ) {
                            gradBiasWeights[ outPlane ] = learningMultiplier * thisBiasChange;
                        }
                    }
                }
            }
        }
    }
    StatefulTimer::instance()->timeCheck(" BackpropWeightsCpu end" );
}

