// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "SquareLossLayer.h"
#include "LossLayer.h"
#include "layer/LayerMaker.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

SquareLossLayer::SquareLossLayer(Layer *previousLayer, SquareLossMaker *maker) :
        LossLayer(previousLayer, maker),
        gradInput(0),
        allocatedSize(0) {
}
VIRTUAL SquareLossLayer::~SquareLossLayer(){
    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string SquareLossLayer::getClassName() const {
    return "SquareLossLayer";
}
VIRTUAL float*SquareLossLayer::getGradInput() {
    return gradInput;
}
VIRTUAL float SquareLossLayer::calcLoss(float const *expected) {
    float loss = 0;
//    float *output = getOutput();
    float *input = previousLayer->getOutput();
//    cout << "SquareLossLayer::calcLoss" << endl;
    int numPlanes = previousLayer->getOutputPlanes();
    int imageSize = previousLayer->getOutputSize();
    int totalLinearSize = batchSize * numPlanes * imageSize * imageSize;
    for(int i = 0; i < totalLinearSize; i++) {
//        if(i < 5) cout << "input[" << i << "]=" << input[i] << endl;
        float diff = input[i] - expected[i];
        float diffSquared = diff * diff;
        loss += diffSquared;
    }
    loss *= 0.5f;
//    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void SquareLossLayer::setBatchSize(int batchSize) {
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    allocatedSize = batchSize;
    gradInput = new float[ batchSize * previousLayer->getOutputNumElements() ];
}
VIRTUAL void SquareLossLayer::calcGradInput(float const*expectedOutput) {
    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < inputNumElements; i++) {
        gradInput[i] = input[i] - expectedOutput[i];
    }
}
VIRTUAL int SquareLossLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL std::string SquareLossLayer::asString() const {
    return "SquareLossLayer{}";
}

