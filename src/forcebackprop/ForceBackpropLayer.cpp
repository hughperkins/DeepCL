// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ForceBackpropLayerMaker.h"

#include "ForceBackpropLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

ForceBackpropLayer::ForceBackpropLayer(Layer *previousLayer, ForceBackpropLayerMaker *maker) :
       Layer(previousLayer, maker),
    outputPlanes(previousLayer->getOutputPlanes()),
    outputSize(previousLayer->getOutputSize()),
    batchSize(0),
    allocatedSize(0),
    output(0) {
}
VIRTUAL ForceBackpropLayer::~ForceBackpropLayer() {
    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string ForceBackpropLayer::getClassName() const {
    return "ForceBackpropLayer";
}
VIRTUAL void ForceBackpropLayer::backward(float learningRate) {
    // do nothing...
}
VIRTUAL float *ForceBackpropLayer::getOutput() {
    return output;
}
VIRTUAL int ForceBackpropLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL bool ForceBackpropLayer::needsBackProp() {
    return true;
}
VIRTUAL void ForceBackpropLayer::printOutput() {
    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
        std::cout << "ForceBackpropLayer n " << n << ":" << std::endl;
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
VIRTUAL void ForceBackpropLayer::print() {
    printOutput();
}
//VIRTUAL bool ForceBackpropLayer::needErrorsBackprop() {
//    return true; // the main reason for this layer :-)
//}
VIRTUAL void ForceBackpropLayer::setBatchSize(int batchSize) {
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
VIRTUAL void ForceBackpropLayer::forward() {
    int totalLinearLength = getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < totalLinearLength; i++) {
        output[i] = input[i];
    }
}
VIRTUAL void ForceBackpropLayer::backward() {
  // do nothing... ?
}
VIRTUAL int ForceBackpropLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL int ForceBackpropLayer::getOutputPlanes() const {
    return outputPlanes;
}
VIRTUAL int ForceBackpropLayer::getOutputCubeSize() const {
    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int ForceBackpropLayer::getOutputNumElements() const {
    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string ForceBackpropLayer::toString() {
    return toString();
}
VIRTUAL std::string ForceBackpropLayer::asString() const {
    return std::string("") + "ForceBackpropLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
}


