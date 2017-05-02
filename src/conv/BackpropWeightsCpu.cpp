// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsCpu.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsCpu::BackpropWeightsCpu(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
}
VIRTUAL BackpropWeightsCpu::~BackpropWeightsCpu() {
}
VIRTUAL void BackpropWeightsCpu::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    gradOutputWrapper->copyToHost();
    imagesWrapper->copyToHost();
    float *gradBias = 0;
    if(dim.biased) {
        gradBiasWrapper->copyToHost();
        gradBias =  (float *)gradBiasWrapper->getHostArray();
    }
    calcGradWeights(batchSize, (float *)gradOutputWrapper->getHostArray(), (float *)imagesWrapper->getHostArray(),
        (float *)gradWeightsWrapper->getHostArray(), gradBias);
    gradWeightsWrapper->copyToDevice();
    if(dim.biased) {
        gradBiasWrapper->copyToDevice();
    }
}
VIRTUAL void BackpropWeightsCpu::calcGradWeights(int batchSize, float *gradOutput,
    float *inputs, float *gradWeights, float *gradBias) {

    StatefulTimer::instance()->timeCheck(" calcGradWeightsCpu start");

    const float learningMultiplier = learningRateToMultiplier(batchSize);

    const int halfFilterSize = dim.filterSize >> 1;
    const int margin = dim.padZeros ? halfFilterSize : 0;
    for(int outPlane = 0; outPlane < dim.numFilters; outPlane++) {
        for(int inputPlane = 0; inputPlane < dim.inputPlanes; inputPlane++) {
            for(int filterRow = 0; filterRow < dim.filterSize; filterRow++) {
                for(int filterCol = 0; filterCol <dim.filterSize; filterCol++) {
                    int weightIndex = (( outPlane
                        * dim.inputPlanes + inputPlane)
                        * dim.filterSize + filterRow)
                        * dim.filterSize + filterCol;
                    float thiswchange = 0;
                    float thisBiasChange = 0;
                    // gradWeights:     [outPlane][inputPlane][filterRow][filterCol]
                    //       aggregate over:  [outRow][outCol][n]
                    for(int outRow = 0; outRow < dim.outputSize; outRow++) {
                        int inputRow = outRow - margin + filterRow;
                        if(inputRow < 0 || inputRow > dim.inputSize - 1) {
                            continue;
                        }
                        for(int outCol = 0; outCol < dim.outputSize; outCol++) {
                            int inputCol = outCol - margin + filterCol;
                            if(inputCol < 0 || inputCol > dim.inputSize - 1) {
                                continue;
                            }
                            for(int n = 0; n < batchSize; n++) {
                                int outputIndex = (( n
                                    * dim.numFilters + outPlane)
                                    * dim.outputSize + outRow)
                                    * dim.outputSize + outCol;
                                float gradOutputValue = gradOutput[outputIndex];
                                int inputIndex = (( n
                                    * dim.inputPlanes + inputPlane)
                                    * dim.inputSize + inputRow)
                                    * dim.inputSize + inputCol;
                                float inputValue = inputs[ inputIndex ];
                                thiswchange += gradOutputValue * inputValue;
                                thisBiasChange += gradOutputValue; // fairly sure this is right.  Fairly :-P
                            }
                        }
                    }
//                    cout << "weight change " << weightIndex << " " << learningMultiplier * thiswchange << endl;
                    gradWeights[ weightIndex ] = thiswchange * learningMultiplier;
                    if(dim.biased) {
                        if(filterRow == margin && filterCol == margin && inputPlane == 0) {
                            gradBias[ outPlane ] = learningMultiplier * thisBiasChange;
                        }
                    }
                }
            }
        }
    }
    StatefulTimer::instance()->timeCheck(" calcGradWeightsCpu end");
}

