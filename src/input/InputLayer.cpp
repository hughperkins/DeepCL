// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "input/InputLayerMaker.h"

#include "input/InputLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

InputLayer::InputLayer(InputLayerMaker *maker) :
       Layer(0, maker),
    batchSize(0),
    allocatedSize(0),
    outputPlanes(maker->_numPlanes),
    outputSize(maker->_imageSize),
    input(0),
    output(0) {
}
VIRTUAL InputLayer::~InputLayer() {
}
VIRTUAL std::string InputLayer::getClassName() const {
    return "InputLayer";
}
VIRTUAL float *InputLayer::getOutput() {
    return output;
}
VIRTUAL bool InputLayer::needsBackProp() {
    return false;
}
VIRTUAL int InputLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL void InputLayer::printOutput() {
    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
        std::cout << "InputLayer n " << n << ":" << std::endl;
        for(int plane = 0; plane < std::min(5, outputPlanes); plane++) {
            if(outputPlanes > 1) std::cout << "    plane " << plane << ":" << std::endl;
            for(int i = 0; i < std::min(5, outputSize); i++) {
                std::cout << "      ";
                for(int j = 0; j < std::min(5, outputSize); j++) {
                    std::cout << getOutput(n, plane, i, j) << " ";
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
VIRTUAL void InputLayer::print() {
    printOutput();
}
 void InputLayer::in(float const*images) {
//        std::cout << "InputLayer::in()" << std::endl;
    this->input = images;
//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
}
VIRTUAL bool InputLayer::needErrorsBackprop() {
    return false;
}
VIRTUAL void InputLayer::setBatchSize(int batchSize) {
//        std::cout << "inputlayer setting batchsize " << batchSize << std::endl;
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[batchSize * getOutputCubeSize() ];
}
VIRTUAL void InputLayer::forward() {
    int totalLinearLength = getOutputNumElements();
    for(int i = 0; i < totalLinearLength; i++) {
        output[i] = input[i];
    }
}
//VIRTUAL void InputLayer::backward(float learningRate, float const *gradOutput) {
//}
VIRTUAL int InputLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL int InputLayer::getOutputPlanes() const {
    return outputPlanes;
}
VIRTUAL int InputLayer::getOutputCubeSize() const {
    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int InputLayer::getOutputNumElements() const {
    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string InputLayer::toString() {
    return asString();
}
VIRTUAL std::string InputLayer::asString() const {
    return std::string("") + "InputLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
}

//template<>VIRTUAL std::string InputLayer<unsigned char>::asString() const {
//    return std::string("") + "InputLayer<unsigned char>{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
//}

//template<>VIRTUAL std::string InputLayer<float>::asString() const {
//    return std::string("") + "InputLayer<float>{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
//}


