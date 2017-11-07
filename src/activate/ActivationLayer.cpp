// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/NeuralNet.h"
#include "util/stringhelper.h"
#include "CppRuntimeBoundary.h"

#include "activate/ActivationLayer.h"
#include "activate/ActivationMaker.h"
#include "activate/ActivationForward.h"
#include "activate/ActivationBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

ActivationLayer::ActivationLayer(EasyCL *cl, Layer *previousLayer, ActivationMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        outputSize(previousLayer->getOutputSize()),
        fn(maker->_activationFunction),
        cl(cl),
        output(0),
        gradInput(0),
        outputWrapper(0),
        gradInputWrapper(0),
//        outputCopiedToHost(false),
//        gradInputCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString(layerIndex) + ": output image size is 0");
    }
    activationForwardImpl = ActivationForward::instance(cl, numPlanes, inputSize, fn);
    activationBackpropImpl = ActivationBackward::instance(cl, numPlanes, inputSize, fn);
}
VIRTUAL ActivationLayer::~ActivationLayer() {
    delete activationForwardImpl;
    delete activationBackpropImpl;
    if(outputWrapper != 0) {
        delete outputWrapper;
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
VIRTUAL std::string ActivationLayer::getClassName() const {
    return "ActivationLayer";
}
VIRTUAL float ActivationLayer::getOutput(int n, int plane, int row, int col) {
    int index = (( n
        * numPlanes + plane)
        * outputSize + row)
        * outputSize + col;
    return output[ index ];
}
VIRTUAL void ActivationLayer::printOutput() {
//    float const*output = getOutput();
//    int outPlanes = getOutputPlanes();
//    int outputNumElements = getOutputSize();
    //std::cout << "  outputs: " << std::endl;
    getOutput();
// output are organized like [imageid][filterid][row][col]
    for(int n = 0; n < std::min(5, batchSize); n++) {
        std::cout << "    n: " << n << std::endl;
        for(int plane = 0; plane < std::min(5, numPlanes); plane++) {
            if(numPlanes > 1) std::cout << "      plane " << plane << std::endl;
            if(outputSize == 1) {
                 std::cout << "        " << getOutput(n, plane, 0, 0) << std::endl;
            } else {
                for(int i = 0; i < std::min(5, outputSize); i++) {
                    std::cout << "      ";
                    for(int j = 0; j < std::min(5, outputSize); j++) {
                        std::cout << getOutput(n, plane, i, j) << " ";
                    }
                    if(outputSize > 5) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if(outputSize > 5) std::cout << " ... " << std::endl;
            }
            if(numPlanes > 5) std::cout << " ... other planes ... " << std::endl;
        }
        if(batchSize > 5) std::cout << " ... other n ... " << std::endl;
    }
}
VIRTUAL void ActivationLayer::setBatchSize(int batchSize) {
//    cout << "ActivationLayer::setBatchSize" << endl;
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
    outputWrapper->createOnDevice();
    gradInput = new float[ previousLayer->getOutputNumElements() ];
    gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), gradInput);
    gradInputWrapper->createOnDevice();
}
VIRTUAL int ActivationLayer::getOutputNumElements() {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *ActivationLayer::getOutput() {
    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
//    cout << "getOutput output[0] " << output[0] << " output[1] " << output[1] << endl;
    return output;
}
VIRTUAL bool ActivationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int ActivationLayer::getOutputNumElements() const {
//    int outputSize = inputSize / poolingSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int ActivationLayer::getOutputCubeSize() const {
    return numPlanes * outputSize * outputSize;
}
VIRTUAL int ActivationLayer::getOutputSize() const {
    return outputSize;
}
VIRTUAL const char *ActivationLayer::getActivationAsCharStar() const {
    return deepcl_stringToCharStar(getActivationFunction()->getName());
}
VIRTUAL int ActivationLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL bool ActivationLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getGradInputWrapper() {
    return gradInputWrapper;
}
VIRTUAL bool ActivationLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL int ActivationLayer::getWeightsSize() const {
    return 0;
}
VIRTUAL int ActivationLayer::getBiasSize() const {
    return 0;
}
VIRTUAL float *ActivationLayer::getGradInput() {
    if(gradInputWrapper->isDeviceDirty()) {
        gradInputWrapper->copyToHost();
//        gradInputCopiedToHost = true;
    }
    return gradInput;
}
VIRTUAL ActivationFunction const *ActivationLayer::getActivationFunction() const {
    return fn;
}
VIRTUAL void ActivationLayer::forward() {
    CLWrapper *inputWrapper = 0;
    if(previousLayer->hasOutputWrapper()) {
        inputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *input = previousLayer->getOutput();
        inputWrapper = cl->wrap(previousLayer->getOutputNumElements(), input);
        inputWrapper->copyToDevice();
    }
    activationForwardImpl->forward(batchSize, inputWrapper, outputWrapper);
//    outputCopiedToHost = false;
    if(!previousLayer->hasOutputWrapper()) {
        delete inputWrapper;
    }
}
VIRTUAL void ActivationLayer::backward() {
    // have no weights to backprop to, just need to backprop the errors

//    CLWrapper *imagesWrapper = 0;
//    if(previousLayer->hasOutputWrapper()) {
//        imagesWrapper = previousLayer->getOutputWrapper();
//    } else {
//        imagesWrapper = cl->wrap(previousLayer->getOutputNumElements(), previousLayer->getOutput());
//        imagesWrapper->copyToDevice();
//    }

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnGradOutputWrapper = false;
    if(nextLayer->providesGradInputWrapper()) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
        gradOutputWrapper->copyToDevice();
        weOwnGradOutputWrapper = true;
    }

    activationBackpropImpl->backward(batchSize, outputWrapper, gradOutputWrapper, gradInputWrapper);
//    gradInputCopiedToHost = false;

//    if(!previousLayer->hasOutputWrapper()) {
//        delete imagesWrapper;
//    }
    if(weOwnGradOutputWrapper) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string ActivationLayer::asString() const {
    return std::string("ActivationLayer{ ") + fn->getDefineName() + " }";
}
VIRTUAL int ActivationLayer::getPersistSize(int version) const {
    // no weights, so:
    return 0;
}

