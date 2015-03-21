// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "PropagateCpu.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

PropagateCpu::PropagateCpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
    {
}
VIRTUAL void PropagateCpu::propagate( int batchSize, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *resultsWrapper ) {
    inputDataWrapper->copyToHost();
    weightsWrapper->copyToHost();
//    weightsWrapper->copyToHost();
  //  biasWeightsWrapper->copyToHost();
    float *biasWeights = 0;
    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
        biasWeights =  (float *)biasWeightsWrapper->getHostArray();
    }
    float *results = propagate( batchSize, (float *)inputDataWrapper->getHostArray(), (float *)weightsWrapper->getHostArray(), biasWeights );
    int resultsSize = batchSize * dim.outputCubeSize;
//        memcpy( (float *)resultsWrapper->getHostArray(), results, sizeof(float) * resultsSize );
    float *hostArray = (float *)resultsWrapper->getHostArray();
    for( int i = 0; i < resultsSize; i++ ) {
        hostArray[i] = results[i];
    }
    resultsWrapper->copyToDevice();
    delete[] results;
}
VIRTUAL float *PropagateCpu::propagate( int batchSize, float *inputData, float *weights, float *biasWeights ) {
//    cout << "PropagateCpu::propagate outputcubesize=" << dim.outputCubeSize << " batchSize=" << batchSize << endl;
    cout << "" << endl;
    float *results = new float[ dim.outputCubeSize * batchSize ];
    for( int n = 0; n < batchSize; n++ ) {
        for( int filter = 0; filter < dim.numFilters; filter++ ) {
            for( int outRow = 0; outRow < dim.outputImageSize; outRow += 1 + dim.skip ) {
                for( int outCol = 0; outCol < dim.outputImageSize; outCol += 1 + dim.skip ) {
                    float sum = 0;
                    for( int inPlane = 0; inPlane < dim.inputPlanes; inPlane++ ) {
//                        cout << "inplane=" << inPlane << endl;
                        for( int u = -dim.halfFilterSize; u <= dim.halfFilterSize; u++ ) {
                            int inRow = outRow * ( dim.skip + 1 ) + u + ( dim.padZeros ? 0 : dim.halfFilterSize );
//                                cout << "candidate inRow " << inRow << endl;
                            if( inRow < 0 || inRow > dim.inputImageSize - 1 ) {
                                continue;
                            }
                            int filterRow = u + dim.halfFilterSize;
                            for( int v = -dim.halfFilterSize; v <= dim.halfFilterSize; v++ ) {
                                int inCol = outCol * ( dim.skip + 1 ) + v + ( dim.padZeros ? 0 : dim.halfFilterSize );
                                int filterCol = v + dim.halfFilterSize;
                                if( inCol < 0 || inCol > dim.inputImageSize - 1 ) {
                                    continue;
                                }
                                int inputIndex = ( ( n
                                    * dim.inputPlanes + inPlane )
                                    * dim.inputImageSize + inRow )
                                    * dim.inputImageSize + inCol;
                                int weightIndex = ( ( filter 
                                    * dim.inputPlanes + inPlane ) 
                                    * dim.filterSize  + filterRow )
                                    * dim.filterSize  + filterCol;
//                                    cout << "inpos " << inRow << "," << inCol << " outpos " << outRow << "," << outCol
//                                        << " filterpos " << filterRow << "," << filterCol << endl;
                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];
                                if( filter == 0 ) {
//                                    cout << "filter[" << filter << "," << inPlane << "," << filterRow << "," << filterCol << "]=" << weights[weightIndex]  << " weightindex=" << weightIndex << endl;
//                                    cout << "input[n=" << n << ",inplane=" << inPlane << "," << inRow << "," << inCol << "]=" << inputData[inputIndex] << endl;

                                    cout << "input[n=" << n << ",inplane=" << inPlane << "," << inRow << "," << inCol << "] input=" << inputData[inputIndex] << " weight=" << weights[weightIndex] << endl;
//                                    cout << "filter[" << filter << "," << inPlane << "," << filterRow << "," << filterCol << "]=" << weights[weightIndex]  << " weightindex=" << weightIndex << endl;
                                }
                                if( sumchange != 0 ) {
//                                        cout << inputData[inputIndex] << " * " << weights[weightIndex] << " = " << sumchange << endl;
                                }
                                sum += sumchange;
//                                cout << "inputIndex=" << inputIndex << " weightIndex=" << weightIndex << 
//                                    "  inputData[inputIndex]=" << inputData[inputIndex] << " weights[weightIndex]=" << weights[weightIndex] << " sumchange " << sumchange << " sum=" << sum << endl;
                            }
                        }
                    }
                    cout << "sum before bias " << sum << endl;
                    if( dim.biased ) {
                        sum += biasWeights[filter];
                    }
                    cout << "sum after bias " << sum << endl;
                    sum = fn->calc( sum );
                    cout << "sum after activate " << sum << endl;
                    int resultsIndex = ( ( n 
                        * dim.numFilters + filter ) 
                        * dim.outputImageSize + outRow )
                        * dim.outputImageSize + outCol;
                    results[resultsIndex] = sum;
                    cout << "resultsIndex=" << resultsIndex << " sum=" << sum << " results[resultsIndex]=" <<
                        results[resultsIndex] << endl;
                }
            }
        }
    }
    return results;
}


