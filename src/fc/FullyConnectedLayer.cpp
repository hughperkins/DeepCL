// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/NeuralNet.h"
#include "fc/FullyConnectedMaker.h"
#include "fc/FullyConnectedLayer.h"
#include "conv/ConvolutionalLayer.h"
#include "conv/ConvolutionalMaker.h"

using namespace std;
using namespace easycl;

#undef VIRTUAL
#define VIRTUAL 

FullyConnectedLayer::FullyConnectedLayer(EasyCL *cl, Layer *previousLayer, FullyConnectedMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes(maker->_numPlanes),
        imageSize(maker->_imageSize),
//        fn(maker->_activationFunction),
        batchSize(0) {
    ConvolutionalMaker *convolutionalMaker = new ConvolutionalMaker();
    convolutionalMaker->numFilters(numPlanes * imageSize * imageSize)
                      ->filterSize(previousLayer->getOutputSize())
                        ->biased(maker->_biased)
                        ->weightsInitializer(maker->_weightsInitializer);
    convolutionalLayer = new ConvolutionalLayer(cl, previousLayer, convolutionalMaker);
//    delete convolutionalMaker;
}

VIRTUAL FullyConnectedLayer::~FullyConnectedLayer() {
    delete convolutionalLayer;
}
VIRTUAL std::string FullyConnectedLayer::getClassName() const {
    return "FullyConnectedLayer";
}
VIRTUAL void FullyConnectedLayer::setBatchSize(int batchSize) {
    convolutionalLayer->previousLayer = this->previousLayer;
    convolutionalLayer->nextLayer = this->nextLayer;
    convolutionalLayer->setBatchSize(batchSize);
    this->batchSize = batchSize;
}
VIRTUAL int FullyConnectedLayer::getOutputCubeSize() const {
    return numPlanes * imageSize * imageSize;
}
VIRTUAL int FullyConnectedLayer::getOutputSize() const {
    return imageSize;
}
VIRTUAL int FullyConnectedLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int FullyConnectedLayer::getPersistSize(int version) const {
    return convolutionalLayer->getPersistSize(version);
}
VIRTUAL void FullyConnectedLayer::persistToArray(int version, float *array) {
    convolutionalLayer->persistToArray(version, array);
}
VIRTUAL void FullyConnectedLayer::unpersistFromArray(int version, float const*array) {
    convolutionalLayer->unpersistFromArray(version, array);
}
VIRTUAL void FullyConnectedLayer::setWeights(float *weights, float *bias) {
    convolutionalLayer->initWeights(weights);
    convolutionalLayer->initBias(bias);
}
VIRTUAL float * FullyConnectedLayer::getWeights() {
    return convolutionalLayer->getWeights();
}
VIRTUAL float * FullyConnectedLayer::getBias() {
    return convolutionalLayer->getBias();
}
VIRTUAL int FullyConnectedLayer::getWeightsSize() const {
    return convolutionalLayer->getWeightsSize();
}
VIRTUAL int FullyConnectedLayer::getBiasSize() const {
    return convolutionalLayer->getBiasSize();
}
VIRTUAL int FullyConnectedLayer::getOutputNumElements() const {
    return convolutionalLayer->getOutputNumElements();
}
VIRTUAL float *FullyConnectedLayer::getOutput() {
    return convolutionalLayer->getOutput();
}
VIRTUAL float *FullyConnectedLayer::getGradInput() {
    return convolutionalLayer->getGradInput();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradWeightsWrapper() {
    return convolutionalLayer->getGradWeightsWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradBiasWrapper() {
    return convolutionalLayer->getGradBiasWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getWeightsWrapper() {
    return convolutionalLayer->getWeightsWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getBiasWrapper() {
    return convolutionalLayer->getBiasWrapper();
}
VIRTUAL bool FullyConnectedLayer::biased() {
    return convolutionalLayer->biased();
}
VIRTUAL bool FullyConnectedLayer::providesGradInputWrapper() const {
    return convolutionalLayer->providesGradInputWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradInputWrapper() {
    return convolutionalLayer->getGradInputWrapper();
}
VIRTUAL bool FullyConnectedLayer::hasOutputWrapper() const {
    return convolutionalLayer->hasOutputWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getOutputWrapper() {
    return convolutionalLayer->getOutputWrapper();
}
//VIRTUAL ActivationFunction const*FullyConnectedLayer::getActivationFunction() {
//    return fn;
//}
VIRTUAL bool FullyConnectedLayer::needsBackProp() {
    return true;;
}
VIRTUAL void FullyConnectedLayer::forward() {
    convolutionalLayer->forward();
}
VIRTUAL void FullyConnectedLayer::backward() {
    convolutionalLayer->backward();
}
VIRTUAL bool FullyConnectedLayer::needsTrainerState() const {
    return true;
}
VIRTUAL TrainerState *FullyConnectedLayer::getTrainerState() {
    return convolutionalLayer->getTrainerState();
}
VIRTUAL TrainerState *FullyConnectedLayer::getBiasTrainerState() {
    return convolutionalLayer->getBiasTrainerState();
}
VIRTUAL void FullyConnectedLayer::setTrainerState(TrainerStateMaker *TrainerStateMaker) {
    convolutionalLayer->setTrainerState(TrainerStateMaker);
}
VIRTUAL std::string FullyConnectedLayer::asString() const {
    return "FullyConnectedLayer{ numPlanes=" + toString(numPlanes) + " imageSize=" + toString(imageSize) + " }";
}

