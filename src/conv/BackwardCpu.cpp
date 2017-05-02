// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackwardCpu.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackwardCpu::BackwardCpu(EasyCL *cl, LayerDimensions dim) :
        Backward(cl, dim)
            {
}
VIRTUAL BackwardCpu::~BackwardCpu() {
}
VIRTUAL float *BackwardCpu::backward(int batchSize, float *inputs,
    float *gradOutput, float *weights) {
    float *gradInput = new float[ batchSize * dim.inputCubeSize ];

//        Timer timer;
    StatefulTimer::instance()->timeCheck("BackwardCpu start");
    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    // handle lower layer...
    // errors for upstream look like [n][inPlane][inRow][inCol]
    // need to aggregate over: [outPlane][outRow][outCol] (?)
    // need to backprop errors along each possible weight
    // each upstream feeds to:
    //    - each of our filters (so numPlanes filters)
    //    - each of our outpoint points (so imageSize * imageSize)
    // for our own backprop, we updated weights for:
    //      [outPlane][inPlane][filterRow][filtercol]
    //    aggregating over: [n][outRow][outCol]
    // errors are provider per [n][inPlane][inRow][inCol]
    for(int n = 0; n < batchSize; n++) {
        for(int upstreamPlane = 0; upstreamPlane < dim.inputPlanes; upstreamPlane++) {
            for(int upstreamRow = 0; upstreamRow < dim.inputSize; upstreamRow++) {
                int minFilterRow = std::max(0, upstreamRow + margin - (dim.outputSize - 1));
                int maxFilterRow = std::min(dim.filterSize - 1, upstreamRow + margin);
                for(int upstreamCol = 0; upstreamCol < dim.inputSize; upstreamCol++) {
                    float sumWeightTimesGradOutput = 0;
                    // aggregate over [outPlane][outRow][outCol]
                    int minFilterCol = std::max(0, upstreamCol + margin - (dim.outputSize -1));
                    int maxFilterCol = std::min(dim.filterSize - 1, upstreamCol + margin);
                    for(int outPlane = 0; outPlane < dim.numFilters; outPlane++) {
                        for(int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
                            int outRow = upstreamRow + margin - filterRow;
                            for(int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                                int outCol = upstreamCol + margin - filterCol;
                                int resultIndex = (( n 
                                    * dim.numFilters + outPlane)
                                    * dim.outputSize + outRow)
                                    * dim.outputSize + outCol;
                                float thisGradOutput = gradOutput[resultIndex];
                                int thisWeightIndex = (( outPlane 
                                    * dim.inputPlanes + upstreamPlane)
                                    * dim.filterSize + filterRow)
                                    * dim.filterSize + filterCol;
                                float thisWeight = weights[thisWeightIndex];
                                sumWeightTimesGradOutput += thisWeight * thisGradOutput;
                            }
                        }
                    }
                    int inputIndex = (( n
                        * dim.inputPlanes + upstreamPlane)
                        * dim.inputSize + upstreamRow)
                        * dim.inputSize + upstreamCol;
                    gradInput[inputIndex] = sumWeightTimesGradOutput; // * activationDerivativeUpstream;
                }
            }
        }
    }
//        timer.timeCheck("calced errors for upstream");   
    StatefulTimer::instance()->timeCheck("BackwardCpu end");

    return gradInput;
}
VIRTUAL void BackwardCpu::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {

    inputDataWrapper->copyToHost();
    gradOutputWrapper->copyToHost();
    weightsWrapper->copyToHost();
//    float *bias = 0;
//    if(dim.biased) {
//        biasWrapper->copyToHost();
//        bias =  (float *)biasWrapper->getHostArray();
//    }
    float *gradInput = backward(batchSize, (float *)inputDataWrapper->getHostArray(),
         (float *)gradOutputWrapper->getHostArray(), (float *)weightsWrapper->getHostArray());
    float *gradInputHostArray = (float*)gradInputWrapper->getHostArray();
    const int gradInputWrapperSize = gradInputWrapper->size();
    for(int i = 0; i < gradInputWrapperSize; i++) {
        gradInputHostArray[i] = gradInput[i];
    }
    gradInputWrapper->copyToDevice();
    delete[] gradInput;
}

