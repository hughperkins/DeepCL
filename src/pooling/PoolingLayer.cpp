// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "PoolingMaker.h"
#include "PoolingLayer.h"
#include "PoolingForward.h"
#include "PoolingBackward.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingLayer::PoolingLayer(EasyCL *cl, Layer *previousLayer, PoolingMaker *maker) :
        Layer(previousLayer, maker),
        padZeros(maker->_padZeros),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        poolingSize(maker->_poolingSize),
        outputSize(maker->_padZeros ? (previousLayer->getOutputSize() + maker->_poolingSize - 1) / maker->_poolingSize : previousLayer->getOutputSize() / maker->_poolingSize),
        cl(cl),
        output(0),
        selectors(0),
        gradInput(0),
        outputWrapper(0),
        selectorsWrapper(0),
        gradInputWrapper(0),
//        outputCopiedToHost(false),
//        gradInputCopiedToHost(false),
        batchSize(0),
        allocatedSize(0){
    if(inputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": output image size is 0");
    }
    poolingForwardImpl = PoolingForward::instance(cl, padZeros, numPlanes, inputSize, poolingSize);
    poolingBackpropImpl = PoolingBackward::instance(cl, padZeros, numPlanes, inputSize, poolingSize);
}
VIRTUAL PoolingLayer::~PoolingLayer() {
    delete poolingForwardImpl;
    delete poolingBackpropImpl;
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(output != 0) {
        delete[] output;
    }
    if(selectorsWrapper != 0) {
        delete selectorsWrapper;
    }
    if(selectors != 0) {
        delete[] selectors;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string PoolingLayer::getClassName() const {
    return "PoolingLayer";
}
VIRTUAL void PoolingLayer::setBatchSize(int batchSize) {
//    cout << "PoolingLayer::setBatchSize" << endl;
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(output != 0) {
        delete[] output;
    }
    if(selectorsWrapper != 0) {
        delete selectorsWrapper;
    }
    if(selectors != 0) {
        delete[] selectors;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[ getOutputNumElements() ];
    outputWrapper = cl->wrap(getOutputNumElements(), output);
    selectors = new int[ getOutputNumElements() ];
    selectorsWrapper = cl->wrap(getOutputNumElements(), selectors);
    gradInput = new float[ previousLayer->getOutputNumElements() ];
    gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), gradInput);
    gradInputWrapper->createOnDevice();
}
VIRTUAL int PoolingLayer::getOutputNumElements() {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *PoolingLayer::getOutput() {
    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool PoolingLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int PoolingLayer::getOutputNumElements() const {
//    int outputSize = inputSize / poolingSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int PoolingLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL int PoolingLayer::getOutputCubeSize() const {
    return numPlanes * outputSize * outputSize;
}
VIRTUAL int PoolingLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int PoolingLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL int PoolingLayer::getPoolingSize() const {
    return poolingSize;
}
VIRTUAL bool PoolingLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getGradInputWrapper() {
    return gradInputWrapper;
}
VIRTUAL bool PoolingLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL float *PoolingLayer::getGradInput() {
    return gradInput;
}
VIRTUAL bool PoolingLayer::getPadZeros() const {
    return padZeros;
}
VIRTUAL ActivationFunction const *PoolingLayer::getActivationFunction() {
    //return previousLayer->getActivationFunction(); // I guess???
    return new LinearActivation();
}
VIRTUAL void PoolingLayer::forward() {
    CLWrapper *upstreamOutputWrapper = 0;
    if(previousLayer->hasOutputWrapper()) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap(previousLayer->getOutputNumElements(), upstreamOutput);
        upstreamOutputWrapper->copyToDevice();
    }
    poolingForwardImpl->forward(batchSize, upstreamOutputWrapper, selectorsWrapper, outputWrapper);
    if(!previousLayer->hasOutputWrapper()) {
        delete upstreamOutputWrapper;
    }

//    cout << "PoolingLayer::forward() selectors after forward: " << endl;
//    for(int i = 0; i < outputSize; i++) {
//        for(int j = 0; j < outputSize; j++) {
//            cout << selectors[ i * outputSize + j ] << " ";
//        }
//        cout << endl;
//    }

//    cout << "PoolingLayer::forward() selectorsWrapper after forward: " << endl;
//    PrintBuffer::printInts(cl, selectorsWrapper, outputSize, outputSize);
}
VIRTUAL void PoolingLayer::backward() {
    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if(nextLayer->providesGradInputWrapper()) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
        gradOutputWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

//    cout << "PoolingLayer::backward selectorsWrapper:" << endl;
//    PrintBuffer::printInts(cl, selectorsWrapper, outputSize, outputSize);

//    int *selectors = reinterpret_cast< int * >(selectorsWrapper->getHostArray());
//    cout << "PoolingLayer::backward selectors before copy to host:" << endl;
//    for(int i = 0; i < outputSize; i++) {
//        for(int j = 0; j < outputSize; j++) {
//            cout << " " << selectors[i * outputSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToHost();
//    cout << "PoolingLayer::backward selectors after copy to host:" << endl;
//    for(int i = 0; i < outputSize; i++) {
//        for(int j = 0; j < outputSize; j++) {
//            cout << " " << selectors[i * outputSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToDevice();

//    selectorsWrapper->copyToHost();

    poolingBackpropImpl->backward(batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper);

//    gradInputWrapper->copyToHost();
//    float *gradInput = reinterpret_cast< float * >(gradInputWrapper->getHostArray());
//    cout << "gradInput:" << endl;
//    for(int i = 0; i < inputSize; i++) {
//        for(int j = 0; j < inputSize; j++) {
////            cout << " " << gradInput[i * inputSize + j];
//            if(gradInput[i * inputSize + j] != 0) {
//                cout << " *";
//            } else {
//                cout << " .";
//            }
//        }
//        cout << endl;
//    }

    if(weOwnErrorsWrapper) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string PoolingLayer::asString() const {
    return "PoolingLayer{ inputPlanes=" + toString(numPlanes) + " inputSize=" + toString(inputSize) + " poolingSize=" + toString(poolingSize) + " }";
}


