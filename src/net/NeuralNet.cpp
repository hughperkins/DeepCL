// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "util/Timer.h"
#include "conv/ConvolutionalLayer.h"
#include "layer/LayerMaker.h"
#include "net/NeuralNetMould.h"
#include "activate/ActivationFunction.h"
#include "util/StatefulTimer.h"
//#include "AccuracyHelper.h"
#include "layer/Layer.h"
#include "input/InputLayer.h"
#include "fc/FullyConnectedLayer.h"
#include "batch/EpochMaker.h"
#include "loss/LossLayer.h"
#include "loss/IAcceptsLabels.h"
#include "util/ExceptionMacros.h"
#include "input/InputLayerMaker.h"
#include "trainers/Trainer.h"
#include "trainers/TrainerMaker.h"
#include "weights/WeightsPersister.h"
#include "CppRuntimeBoundary.h"

#include "net/NeuralNet.h"

using namespace std;
using namespace easycl;

//static std::mt19937 random;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

NeuralNet::NeuralNet(EasyCL *cl) :
        cl(cl) {
    trainer = 0;
    isTraining = true;
}
STATIC NeuralNet *NeuralNet::instance(EasyCL *cl) {
    return new NeuralNet(cl);
}
STATIC NeuralNet *NeuralNet::instance(EasyCL *cl, int numPlanes, int imageSize) {
    return new NeuralNet(cl, numPlanes, imageSize);
}
STATIC NeuralNet *NeuralNet::instance3(EasyCL *cl, int numPlanes, int imageSize) {
    return new NeuralNet(cl, numPlanes, imageSize);
}
void NeuralNet::deleteMe() {
    delete this;
}
/// Constructor
NeuralNet::NeuralNet(EasyCL *cl, int numPlanes, int imageSize) :
        cl(cl) {
    addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    trainer = 0;
}
NeuralNet::~NeuralNet() {
    for(int i = 0; i < (int)layers.size(); i++) {
        delete layers[i];
    }
}
STATIC NeuralNetMould *NeuralNet::maker(EasyCL *cl) {
    return new NeuralNetMould(cl);
}
NeuralNet *NeuralNet::clone() {
    NeuralNet *copy = new NeuralNet(cl);
    for(vector<Layer *>::iterator it = layers.begin(); it != layers.end(); it++) {
        LayerMaker2 *maker = (*it)->maker;

        LayerMaker2 *makerCopy = maker->clone();
        copy->addLayer(makerCopy);
    }
    copy->print();
    cout << "outputimagesize: " << copy->getOutputSize() << endl;
    return copy;
}
EasyCL *NeuralNet::getCl() {
    return cl;
}
/// Add a network layer, using a LayerMaker2 object
PUBLICAPI void NeuralNet::addLayer(LayerMaker2 *maker) {
//    cout << "neuralnet::insert numplanes " << inputLayerMaker._numPlanes << " imageSize " << inputLayerMaker._imageSize << endl;
    maker->setCl(cl);
    Layer *layer = maker->createLayer(getLastLayer());
    layers.push_back(layer);
}
PUBLICAPI void NeuralNet::initWeights(int layerIndex, float *weights, float *bias) {
    initWeights(layerIndex, weights);
    initBias(layerIndex, bias);
}
PUBLICAPI void NeuralNet::initWeights(int layerIndex, float *weights) {
    layers[layerIndex]->initWeights(weights);
}
PUBLICAPI void NeuralNet::initBias(int layerIndex, float *weights) {
    layers[layerIndex]->initBias(weights);
}
/// \brief calculate the loss, based on the passed in expectedValues array
///
/// \publicapi
///
/// Calculate the loss, based on the passed in expectedValues array
/// which should be the same size as the output of the final layer
/// of the network
PUBLICAPI float NeuralNet::calcLoss(float const *expectedValues) {
    return dynamic_cast<LossLayer*>(getLastLayer())->calcLoss(expectedValues);
}
PUBLICAPI float NeuralNet::calcLossFromLabels(int const *labels) {
    return dynamic_cast<IAcceptsLabels*>(getLastLayer())->calcLossFromLabels(labels);
}
float NeuralNet::calcLoss(OutputData *outputData) {
    return dynamic_cast<LossLayer*>(getLastLayer())->calcLoss(outputData);
}
int NeuralNet::calcNumRight(OutputData *outputData) {
    return dynamic_cast<LossLayer*>(getLastLayer())->calcNumRight(outputData);
}
EpochMaker *NeuralNet::epochMaker(Trainer *trainer) {
     return new EpochMaker(this, trainer);
}
VIRTUAL LossLayerMaker *NeuralNet::cloneLossLayerMaker() const {
    LossLayer const *lossLayer = dynamic_cast< LossLayer const*>(getLastLayer());
    if(lossLayer == 0) {
        throw runtime_error("error: last layer must be a losslayer");
    }
    return dynamic_cast< LossLayerMaker *>(lossLayer->maker->clone());
//    throw runtime_error("need to implement neuralnet::clonelosslayermaker :-)");
//    LossLayer const*lossLayer = dynamic_cast< LossLayer const*>(getLastLayer());
//    return dynamic_cast< LossLayerMaker *>(lossLayer->maker->clone(clonePreviousLayer) ) ;
}
PUBLICAPI InputLayer *NeuralNet::getFirstLayer() {
    return dynamic_cast<InputLayer *>(layers[0]);
}
PUBLICAPI Layer *NeuralNet::getLastLayer() {
    if(layers.size() == 0) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI int NeuralNet::getNumLayers() const {
    return (int)layers.size();
}
PUBLICAPI Layer *NeuralNet::getLayer(int index) {
    if(layers.size() == 0) {
        return 0;
    }
    if(index < 0 || index > (int)layers.size() - 1) {
        return 0;
    }
    return layers[index];
}
PUBLICAPI Layer const*NeuralNet::getLastLayer() const {
    if(layers.size() == 0) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputPlanes() const {
    return getLastLayer()->getOutputPlanes();
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputSize() const {
    return getLastLayer()->getOutputSize();
}
PUBLICAPI void NeuralNet::setBatchSize(int batchSize) {
    for(std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
        (*it)->setBatchSize(batchSize);
    }
}
PUBLICAPI void NeuralNet::setTraining(bool training) {
    for(std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
        (*it)->setTraining(training);
    }
}
PUBLICAPI int NeuralNet::calcNumRight(int const *labels) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if(acceptsLabels == 0) {
        THROW("You need to add a IAcceptsLabels as the last layer, in order to use calcNumRight");
    }
    return acceptsLabels->calcNumRightFromLabels(labels);
}
PUBLICAPI void NeuralNet::forward(float const*images) {
    // forward...
    dynamic_cast<InputLayer *>(layers[0])->in(images);
    for(int layerId = 0; layerId < (int)layers.size(); layerId++) {
        StatefulTimer::setPrefix("layer" + toString(layerId) + " ");
        layers[layerId]->forward();
        StatefulTimer::setPrefix("");
    }
}
/// \brief note: this does no learning, just calculates the gradients
PUBLICAPI void NeuralNet::backwardFromLabels(int const *labels) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if(acceptsLabels == 0) {
        throw std::runtime_error("Must add a child of IAcceptsLabels as last layer, to use backwardFromLabels");
    }
    acceptsLabels->calcGradInputFromLabels(labels);
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        Layer *layer = layers[layerIdx];
        if(layer->needsBackProp()) {
            layer->backward();
        }
        StatefulTimer::setPrefix("");
    }
}
/// \brief note: this does no learning, just calculates the gradients
PUBLICAPI void NeuralNet::backward(float const *expectedOutput) {
    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    if(lossLayer == 0) {
        throw std::runtime_error("Must add a LossLayer as last layer of net");
    }
    lossLayer->calcGradInput(expectedOutput);
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        layers[layerIdx]->backward();
        StatefulTimer::setPrefix("");
    }
}
void NeuralNet::backward(OutputData *outputData) {
    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    lossLayer->calcGradInput(outputData);
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer
        Layer *layer = getLayer(layerIdx);
        if(!layer->needsBackProp()) {
            break;
        }
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        layer->backward();
        StatefulTimer::setPrefix("");
    }
}
PUBLICAPI int NeuralNet::getNumLayers() {
    return (int)layers.size();
}
PUBLICAPI float const *NeuralNet::getOutput(int layer) const {
    return layers[layer]->getOutput();
}
PUBLICAPI int NeuralNet::getInputCubeSize() const {
    return layers[ 0 ]->getOutputCubeSize();
}
PUBLICAPI int NeuralNet::getOutputCubeSize() const {
    return layers[ layers.size() - 1 ]->getOutputCubeSize();
}
PUBLICAPI float const *NeuralNet::getOutput() const {
    return getOutput((int)layers.size() - 1);
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputNumElements() const {
    return getLastLayer()->getOutputNumElements();
}
void NeuralNet::print() {
    cout << this->asString();
    printParamStats();
//    int i = 0; 
//    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
//        std::cout << "layer " << i << ":" << (*it)->asString() << endl;
//        i++;
//    }
}
void NeuralNet::printWeights() {
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printWeights();
        i++;
    }
}
void NeuralNet::printOutput() {
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printOutput();
        i++;
    }
}
VIRTUAL void NeuralNet::setTrainer(Trainer *trainer) {
    this->trainer = trainer;
}
void NeuralNet::printParamStats() {
    int sum = 0;
    int skip = 0;
    int precision = (int)std::cout.precision();
//    cout << "precision: " << precision << endl;
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
        int size = (*it)->getPersistSize(WeightsPersister::latestVersion);
        sum += size;
        if(! size){
            skip++;
        }
    }
    std::cout << "Parameters overview: (skipping " << skip << " layers with 0 params)" << std::endl;
    int i = 0;
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++, i++) {
        int size = (*it)->getPersistSize(WeightsPersister::latestVersion);
        if(size) {
            std::cout << "layer " << i << ": params=" << size << "\t";
            std::cout << std::fixed << std::setprecision(1) << ((float) 100 * size)/sum << "%";
            std::cout << std::endl;
        }
    }
    if(i){
        std::cout << "TOTAL  : params=" << sum << std::endl;
    }
    // reset the cout properties, so that I dont spend 2 hours figuring out why my weights
    // all changed to 0.0 and 0.1 :-P
    std::cout << setprecision(precision);
    std::cout.unsetf(ios_base::floatfield);
}
PUBLICAPI std::string NeuralNet::asString() {
    std::string result = "";
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
        result += "layer " + toString(i) + ":" + (*it)->asString() + "\n";
        i++;
    }    
    return result;
}
PUBLICAPI const char * NeuralNet::asNewCharStar() { // call deepcl_deleteCharStar to delete this
    std::string result = "";
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
        result += "layer " + toString(i) + ":" + (*it)->asString() + "\n";
        i++;
    }
    return deepcl_stringToCharStar(result);
}

