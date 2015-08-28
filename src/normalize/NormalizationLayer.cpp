// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "normalize/NormalizationLayerMaker.h"

#include "normalize/NormalizationLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

NormalizationLayer::NormalizationLayer(Layer *previousLayer, NormalizationLayerMaker *maker) :
       Layer(previousLayer, maker),
    translate(maker->_translate),
    scale(maker->_scale),
    outputPlanes(previousLayer->getOutputPlanes()),
    outputSize(previousLayer->getOutputSize()),
    batchSize(0),
    allocatedSize(0),
    output(0) {
}
VIRTUAL NormalizationLayer::~NormalizationLayer() {
    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string NormalizationLayer::getClassName() const {
    return "NormalizationLayer";
}
VIRTUAL float *NormalizationLayer::getOutput() {
    return output;
}
VIRTUAL ActivationFunction const *NormalizationLayer::getActivationFunction() {
    return new LinearActivation();
}
VIRTUAL int NormalizationLayer::getPersistSize(int version) const {
    if(version == 1) {
        return 0;
    }
    return 2;
}
VIRTUAL void NormalizationLayer::persistToArray(int version, float *array) {
    if(version == 1) {
        return;
    }
    array[0] = translate;
    array[1] = scale;
}
/// \brief initialize the current weights and biases from array
VIRTUAL void NormalizationLayer::unpersistFromArray(int version, float const*array) {
    if(version == 1) {
        return;
    }
    translate = array[0];
    scale = array[1];
}
VIRTUAL bool NormalizationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL void NormalizationLayer::printOutput() const {
    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
        std::cout << "NormalizationLayer n " << n << ":" << std::endl;
        for(int plane = 0; plane < std::min(5, outputPlanes); plane++) {
            if(outputPlanes > 1) std::cout << "    plane " << plane << ":" << std::endl;
            for(int i = 0; i < std::min(5, outputSize); i++) {
                std::cout << "      ";
                for(int j = 0; j < std::min(5, outputSize); j++) {
                    std::cout << getResult(n, plane, i, j) << " ";
//output[
//                            n * numPlanes * imageSize*imageSize +
//                            plane*imageSize*imageSize +
//                            i * imageSize +
//                            j ] << " ";
                }
                if(outputSize > 5) std::cout << " ... ";
                std::cout << std::endl;
            }
            if(outputSize > 5) std::cout << " ... " << std::endl;
        }
        if(outputPlanes > 5) std::cout << "   ... other planes ... " << std::endl;
    }
    if(batchSize > 5) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void NormalizationLayer::print() const {
    printOutput();
}
VIRTUAL bool NormalizationLayer::needErrorsBackprop() {
    return false;
}
VIRTUAL void NormalizationLayer::setBatchSize(int batchSize) {
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    this->batchSize = batchSize;
    this->allocatedSize = allocatedSize;
    output = new float[ getOutputNumElements() ];
}
VIRTUAL void NormalizationLayer::forward() {
    int totalLinearLength = getOutputNumElements();
    float *upstreamOutput = previousLayer->getOutput();
    for(int i = 0; i < totalLinearLength; i++) {
        output[i] = (upstreamOutput[i] + translate) * scale;
    }
}
VIRTUAL void NormalizationLayer::backward(float learningRate, float const *gradOutput) {
  // do nothing...
}
VIRTUAL int NormalizationLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL int NormalizationLayer::getOutputPlanes() const {
    return outputPlanes;
}
VIRTUAL int NormalizationLayer::getOutputCubeSize() const {
    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int NormalizationLayer::getOutputNumElements() const {
    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string NormalizationLayer::toString() {
    return toString();
}
VIRTUAL std::string NormalizationLayer::asString() const {
    return std::string("") + "NormalizationLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " translate=" + ::toString(translate) + 
        " scale=" + ::toString(scale) + " }";
}


