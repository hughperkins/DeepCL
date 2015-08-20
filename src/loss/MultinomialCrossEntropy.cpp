// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "MultinomialCrossEntropy.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

MultinomialCrossEntropy::MultinomialCrossEntropy(Layer *previousLayer, MultinomialCrossEntropyMaker const*maker) :
        LossLayer(previousLayer, maker),
        errors(0),
        allocatedSize(0) {
}
VIRTUAL MultinomialCrossEntropy::~MultinomialCrossEntropy(){
    if(errors != 0) {
        delete[] errors;
    }
}
VIRTUAL float*MultinomialCrossEntropy::getGradInput() {
    return errors;
}
VIRTUAL float MultinomialCrossEntropy::calcLoss(float const *expected) {
    float loss = 0;
    float *output = getOutput();
//    cout << "MultinomialCrossEntropy::calcLoss" << endl;
    // this is matrix subtraction, then element-wise square, then aggregation
    int numPlanes = previousLayer->getOutputPlanes();
    int imageSize = previousLayer->getOutputSize();
    for(int imageId = 0; imageId < batchSize; imageId++) {
        for(int plane = 0; plane < numPlanes; plane++) {
            for(int outRow = 0; outRow < imageSize; outRow++) {
                for(int outCol = 0; outCol < imageSize; outCol++) {
                    int resultOffset = (( imageId
                         * numPlanes + plane)
                         * imageSize + outRow)
                         * imageSize + outCol;
 //                   int resultOffset = getResultIndex(imageId, plane, outRow, outCol); //imageId * numPlanes + out;
                    float expectedOutput = expected[resultOffset];
                    float actualOutput = output[resultOffset];
                    float negthisloss = expectedOutput * log(actualOutput);
                    loss -= negthisloss;
                }
            }
        }            
    }
    loss *= 0.5f;
//    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void MultinomialCrossEntropy::setBatchSize(int batchSize) {
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(errors != 0) {
        delete[] errors;
    }
    errors = new float[ batchSize * previousLayer->getOutputNumElements() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
// just do naively for now, then add sigmoid short-cutting later
VIRTUAL void MultinomialCrossEntropy::calcGradInput(float const*expectedOutput) {
    ActivationFunction const*fn = previousLayer->getActivationFunction();
    int outputNumElements = previousLayer->getOutputNumElements();
    float *output = previousLayer->getOutput();
    for(int i = 0; i < outputNumElements; i++) {
        float result = output[i];
        float partialOutBySum = fn->calcDerivative(result);
        float partialLossByOut = - expectedOutput[i] / result;
        errors[i] = partialLossByOut * partialOutBySum;
    }
}

