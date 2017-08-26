// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "dropout/DropoutLayer.h"
#include "dropout/DropoutMaker.h"
#include "dropout/DropoutForward.h"
#include "dropout/DropoutBackward.h"
#include "util/RandomSingleton.h"
#include "clmath/MultiplyBuffer.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutLayer::DropoutLayer(EasyCL *cl, Layer *previousLayer, DropoutMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        dropRatio(maker->_dropRatio),
        outputSize(previousLayer->getOutputSize()),
        random(RandomSingleton::instance()),
        cl(cl),
        masks(0),
        output(0),
        gradInput(0),
        maskWrapper(0),
        outputWrapper(0),
        gradInputWrapper(0),
//        outputCopiedToHost(false),
//        gradInputCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString(layerIndex) + ": output image size is 0");
    }
    dropoutForwardImpl = DropoutForward::instance(cl, numPlanes, inputSize, dropRatio);
    dropoutBackwardImpl = DropoutBackward::instance(cl, numPlanes, inputSize, dropRatio);
    multiplyBuffer = new MultiplyBuffer(cl);
}
VIRTUAL DropoutLayer::~DropoutLayer() {
    delete multiplyBuffer;
    delete dropoutForwardImpl;
    delete dropoutBackwardImpl;
    if(maskWrapper != 0) {
        delete maskWrapper;
    }
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(masks != 0) {
        delete[] masks;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string DropoutLayer::getClassName() const {
    return "DropoutLayer";
}
VIRTUAL void DropoutLayer::fortesting_setRandomSingleton(RandomSingleton *random) {
    this->random = random;
}
VIRTUAL void DropoutLayer::setBatchSize(int batchSize) {
//    cout << "DropoutLayer::setBatchSize" << endl;
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(maskWrapper != 0) {
        delete maskWrapper;
    }
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(masks != 0) {
        delete[] masks;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    masks = new unsigned char[ getOutputNumElements() ];
    generateMasks();
    maskWrapper = cl->wrap(getOutputNumElements(), masks);
    output = new float[ getOutputNumElements() ];
    outputWrapper = cl->wrap(getOutputNumElements(), output);
    outputWrapper->createOnDevice();
    gradInput = new float[ previousLayer->getOutputNumElements() ];
    gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), gradInput);
    gradInputWrapper->createOnDevice();
}
VIRTUAL int DropoutLayer::getOutputNumElements() {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *DropoutLayer::getOutput() {
    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool DropoutLayer::needsBackProp() {
    return previousLayer->needsBackProp(); // seems highly unlikely that we wouldnt have to backprop
                                           // but anyway, we dont have any weights ourselves
                                           // so just depends on upstream
}
VIRTUAL int DropoutLayer::getOutputNumElements() const {
//    int outputSize = inputSize / dropoutSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int DropoutLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL int DropoutLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int DropoutLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL bool DropoutLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getGradInputWrapper() {
    return gradInputWrapper;
}
VIRTUAL bool DropoutLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL float *DropoutLayer::getGradInput() {
    return gradInput;
}
VIRTUAL ActivationFunction const *DropoutLayer::getActivationFunction() {
    return new LinearActivation();
}
//VIRTUAL void DropoutLayer::generateMasks() {
//    int totalInputLinearSize = getOutputNumElements();
////    int numBytes = (totalInputLinearSize+8-1)/8;
////    unsigned char *bitsField = new unsigned char[numBytes];
//    int idx = 0;
//    unsigned char thisByte = 0;
//    int bitsPacked = 0;
//    for(int i = 0; i < totalInputLinearSize; i++) {
//        //double value = ((int)random() % 10000) / 20000.0f + 0.5f;
//        // 1 means we pass value through, 0 means we drop
//        // dropRatio is probability that mask value is 0 therefore
//        // so higher dropRatio => more likely to be 0
//        unsigned char bit = random->_uniform() <= dropRatio ? 0 : 1;
////        unsigned char bit = 0;
//        thisByte <<= 1;
//        thisByte |= bit;
//        bitsPacked++;
//        if(bitsPacked >= 8) {
//            masks[idx] = thisByte;
//            idx++;
//            bitsPacked = 0;
//        }
//    }
//}
VIRTUAL void DropoutLayer::generateMasks() {
    int totalInputLinearSize = getOutputNumElements();
    for(int i = 0; i < totalInputLinearSize; i++) {
        masks[i] = random->_uniform() <= dropRatio ? 0 : 1;
    }
}
VIRTUAL void DropoutLayer::forward() {
    CLWrapper *upstreamOutputWrapper = 0;
    if(previousLayer->hasOutputWrapper()) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap(previousLayer->getOutputNumElements(), upstreamOutput);
        upstreamOutputWrapper->copyToDevice();
    }

//    cout << "training: " << training << endl;
    if(training) {
        // create new masks...
        generateMasks();
        maskWrapper->copyToDevice();
        dropoutForwardImpl->forward(batchSize, maskWrapper, upstreamOutputWrapper, outputWrapper);
    } else {
        // if not training, then simply skip the dropout bit, copy the buffers directly
        multiplyBuffer->multiply(getOutputNumElements(), dropRatio, upstreamOutputWrapper, outputWrapper);
    }
    if(!previousLayer->hasOutputWrapper()) {
        delete upstreamOutputWrapper;
    }
}
VIRTUAL void DropoutLayer::backward() {
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
    maskWrapper->copyToDevice();
    dropoutBackwardImpl->backward(batchSize, maskWrapper, gradOutputWrapper, gradInputWrapper);
    if(weOwnErrorsWrapper) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string DropoutLayer::asString() const {
    return "DropoutLayer{ dropRatio=" + toString(dropRatio) + " }";
}


