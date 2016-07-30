// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "RandomTranslations.h"
#include "RandomTranslationsMaker.h"
#include "util/RandomSingleton.h"
#include "Translator.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

RandomTranslations::RandomTranslations(Layer *previousLayer, RandomTranslationsMaker *maker) :
        Layer(previousLayer, maker),
        translateSize(maker->_translateSize),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        outputSize(previousLayer->getOutputSize()),
        output(0),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": output image size is 0");
    }
    if(previousLayer->needsBackProp()) {
        throw runtime_error("Error: RandomTranslations layer does not provide backprop currently, so you cannot put it after a layer that needs backprop");
    }
}
VIRTUAL RandomTranslations::~RandomTranslations() {
    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string RandomTranslations::getClassName() const {
    return "RandomTranslations";
}
VIRTUAL void RandomTranslations::setBatchSize(int batchSize) {
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[ getOutputNumElements() ];
}
VIRTUAL int RandomTranslations::getOutputNumElements() {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *RandomTranslations::getOutput() {
    return output;
}
VIRTUAL bool RandomTranslations::needsBackProp() {
    return false;
}
VIRTUAL int RandomTranslations::getOutputNumElements() const {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int RandomTranslations::getOutputSize() const {
    return outputSize;
}
VIRTUAL int RandomTranslations::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int RandomTranslations::getTranslationSize() const {
    return translateSize;
}
VIRTUAL int RandomTranslations::getOutputCubeSize() const {
    return numPlanes * outputSize * outputSize * batchSize;
}
VIRTUAL int RandomTranslations::getPersistSize(int version) const {
    return 0;
}
VIRTUAL bool RandomTranslations::providesGradInputWrapper() const {
    return false;
}
VIRTUAL bool RandomTranslations::hasOutputWrapper() const {
    return false;
}
VIRTUAL void RandomTranslations::forward() {
    float *upstreamOutput = previousLayer->getOutput();
    if(!training) {
        memcpy(output, upstreamOutput, sizeof(float) * getOutputNumElements());
        return;
    }
    for(int n = 0; n < batchSize; n++) {
        const int translateRows = RandomSingleton::instance()->uniformInt(- translateSize, translateSize);
        const int translateCols = RandomSingleton::instance()->uniformInt(- translateSize, translateSize);
        Translator::translate(n, numPlanes, inputSize, translateRows, translateCols, upstreamOutput, output);
    }
}
VIRTUAL std::string RandomTranslations::asString() const {
    return "RandomTranslations{ inputPlanes=" + toString(numPlanes) + " inputSize=" + toString(inputSize) + " translateSize=" + toString(translateSize) + " }";
}


