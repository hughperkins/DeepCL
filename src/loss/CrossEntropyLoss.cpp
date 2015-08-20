// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "CrossEntropyLoss.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

CrossEntropyLoss::CrossEntropyLoss(Layer *previousLayer, CrossEntropyLossMaker *maker) :
        LossLayer(previousLayer, maker),
        gradInput(0),
        allocatedSize(0) {
}
VIRTUAL CrossEntropyLoss::~CrossEntropyLoss(){
    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string CrossEntropyLoss::getClassName() const {
    return "CrossEntropyLoss";
}
VIRTUAL float*CrossEntropyLoss::getGradInput() {
    return gradInput;
}
VIRTUAL int CrossEntropyLoss::getPersistSize(int version) const {
    return 0;
}
VIRTUAL float CrossEntropyLoss::calcLoss(float const *expected) {
    float loss = 0;
    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
//    cout << "CrossEntropyLoss::calcLoss" << endl;
    for(int i = 0; i < inputNumElements; i++) {
        float expectedOutput = expected[i];
        float inputValue = input[i];
        float negthisloss = expectedOutput * log(inputValue) 
            + (1 - expectedOutput) * log(1 - inputValue);
        loss -= negthisloss;
    }
    return loss;
 }
VIRTUAL void CrossEntropyLoss::setBatchSize(int batchSize) {
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    gradInput = new float[ batchSize * previousLayer->getOutputNumElements() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
// just do naively for now, then add sigmoid short-cutting later
VIRTUAL void CrossEntropyLoss::calcGradInput(float const*expectedOutput) {
    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < inputNumElements; i++) {
        gradInput[i] = (input[i] - expectedOutput[i]) / input[i] / (1.0f - input[i]);
    }
}

