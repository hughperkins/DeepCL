// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "Timer.h"
#include "ConvolutionalLayer.h"
#include "LayerMaker.h"
#include "ActivationFunction.h"
#include "StatefulTimer.h"
#include "AccuracyHelper.h"
#include "NeuralNetMould.h"
#include "Layer.h"
#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "EpochMaker.h"
#include "LossLayer.h"
#include "IAcceptsLabels.h"
#include "ExceptionMacros.h"

#include "NeuralNet.h"

using namespace std;

//static std::mt19937 random;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

NeuralNet::NeuralNet( int numPlanes, int boardSize ) {
    cl = new OpenCLHelper();
    InputLayerMaker *maker = new InputLayerMaker( this, numPlanes, boardSize );
    maker->insert();
}
NeuralNet::~NeuralNet() {
    for( int i = 0; i < layers.size(); i++ ) {
        delete layers[i];
    }
    delete cl;
}
OpenCLHelper *NeuralNet::getCl() {
    return cl;
}
STATIC NeuralNetMould *NeuralNet::maker() {
    return new NeuralNetMould();
}
FullyConnectedMaker *NeuralNet::fullyConnectedMaker() {
    return new FullyConnectedMaker( this, getLastLayer() );
}
ConvolutionalMaker *NeuralNet::convolutionalMaker() {
    return new ConvolutionalMaker( this, getLastLayer() );
}
PoolingMaker *NeuralNet::poolingMaker() {
    return new PoolingMaker( this, getLastLayer() );
}
NormalizationLayerMaker *NeuralNet::normalizationMaker() {
    return new NormalizationLayerMaker( this, getLastLayer() );
}
SquareLossMaker *NeuralNet::squareLossMaker() {
    return new SquareLossMaker( this, getLastLayer() );
}
CrossEntropyLossMaker *NeuralNet::crossEntropyLossMaker() {
    return new CrossEntropyLossMaker( this, getLastLayer() );
}
SoftMaxMaker *NeuralNet::softMaxLossMaker() {
    return new SoftMaxMaker( this, getLastLayer() );
}
void NeuralNet::initWeights( int layerIndex, float *weights, float *biasWeights ) {
    initWeights( layerIndex, weights );
    initBiasWeights( layerIndex, biasWeights );
}
void NeuralNet::initWeights( int layerIndex, float *weights ) {
    layers[layerIndex]->initWeights( weights );
}
void NeuralNet::initBiasWeights( int layerIndex, float *weights ) {
    layers[layerIndex]->initBiasWeights( weights );
}
void NeuralNet::printWeightsAsCode() {
    for( int layer = 1; layer < layers.size(); layer++ ) {
        layers[layer]->printWeightsAsCode();
    }
}
void NeuralNet::printBiasWeightsAsCode() {
    for( int layer = 1; layer < layers.size(); layer++ ) {
        layers[layer]->printBiasWeightsAsCode();
    }
}
float NeuralNet::calcLoss(float const *expectedValues ) {
    return dynamic_cast<LossLayer*>(layers[layers.size()-1])->calcLoss( expectedValues );
}
float NeuralNet::calcLossFromLabels(int const *labels ) {
    return dynamic_cast<IAcceptsLabels*>(layers[layers.size()-1])->calcLossFromLabels( labels );
}
EpochMaker *NeuralNet::epochMaker() {
     return new EpochMaker(this);
}
InputLayer *NeuralNet::getFirstLayer() {
    return dynamic_cast<InputLayer*>( layers[0] );
}
Layer *NeuralNet::getLastLayer() {
    return layers[layers.size() - 1];
}
Layer *NeuralNet::addLayer( LayerMaker *maker ) {
    Layer *previousLayer = 0;
    if( layers.size() > 0 ) {
        previousLayer = layers[ layers.size() - 1 ];
    }
//    maker->setPreviousLayer( previousLayer );
    Layer *layer = maker->instance();
    layers.push_back( layer );
    return layer;
}
void NeuralNet::setBatchSize( int batchSize ) {
    for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        (*it)->setBatchSize( batchSize );
    }
}
float NeuralNet::doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels ) {
    return doEpochFromLabels( learningRate, batchSize, numImages, images, labels, 0 );
}
float NeuralNet::doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels, int *p_totalCorrect ) {
//        Timer timer;
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if( acceptsLabels == 0 ) {
        THROW("You need to add a IAcceptsLabels as the last layer, in order to use doEpochFromLabels");
    }
    setBatchSize( batchSize );
    int numBatches = ( numImages + batchSize - 1 ) / batchSize;
    float loss = 0;
    int total = 0;
    int numRight = 0;
    int numLabelsPerExample = acceptsLabels->getNumLabelsPerExample();
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = numImages - batchStart;  // eg, we have 5 images, and batchsize is 3
                                                         // so last batch size is: 2 = 5 - 3
            setBatchSize( thisBatchSize );
        }
        learnBatchFromLabels( learningRate, &(images[batchStart*getInputCubeSize()]), &(labels[batchStart * numLabelsPerExample]) );
        loss += acceptsLabels->calcLossFromLabels( &(labels[batchStart * numLabelsPerExample]) );
        if( p_totalCorrect != 0 ) {
            numRight += acceptsLabels->calcNumRight( &(labels[batchStart * numLabelsPerExample]) );
        }
    }
    if( p_totalCorrect != 0 ) {
        *p_totalCorrect = numRight;
    }
    return loss;
}
float NeuralNet::doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults ) {
//        Timer timer;
    if( dynamic_cast<LossLayer*>(getLastLayer()) == 0 ) {
        THROW("You need to add a LossLayer as the last layer of the network");
    }
    setBatchSize( batchSize );
    int numBatches = ( numImages + batchSize - 1 ) / batchSize;
    float loss = 0;
    int total = 0;
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = numImages - batchStart;  // eg, we have 5 images, and batchsize is 3
                                                         // so last batch size is: 2 = 5 - 3
            setBatchSize( thisBatchSize );
        }
//            std::cout << " batch " << batch << " start " << batchStart << " inputsizeperex " << getInputSizePerExample() <<
//             " resultssizeperex " << getResultsSizePerExample() << std::endl;
        learnBatch( learningRate, &(images[batchStart*getInputCubeSize()]), &(expectedResults[batchStart*getOutputCubeSize()]) );
        loss += calcLoss( &(expectedResults[batchStart*getOutputCubeSize()]) );
    }
    return loss;
}
int NeuralNet::calcNumRight( int const *labels ) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if( acceptsLabels == 0 ) {
        THROW("You need to add a IAcceptsLabels as the last layer, in order to use calcNumRight");
    }
    return acceptsLabels->calcNumRight( labels );
}
float NeuralNet::doEpochWithCalcTrainingAccuracy( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults, int const *labels, int *p_totalCorrect ) {
    setBatchSize( batchSize );
    int numBatches = ( numImages + batchSize - 1 ) / batchSize;
    std::cout << "numBatches: " << numBatches << std::endl;
    float loss = 0;
    int numRight = 0;
    int total = 0;
    if( getLastLayer()->getOutputBoardSize() != 1 ) {
        THROW("Last layer should have board size of 1, and number of planes equal number of categories, if you want to measure training accuracy");
    }
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = numImages - batchStart;  // eg, we have 5 images, and batchsize is 3
                                                         // so last batch size is: 2 = 5 - 3
            setBatchSize( thisBatchSize );
        }
//            std::cout << " batch " << batch << " start " << batchStart << " inputsizeperex " << getInputSizePerExample() <<
//             " resultssizeperex " << getResultsSizePerExample() << std::endl;
        learnBatch( learningRate, &(images[batchStart*getInputCubeSize()]), &(expectedResults[batchStart*getOutputCubeSize()]) );
        StatefulTimer::timeCheck("after batch forward-backward prop");
        numRight += AccuracyHelper::calcNumRight( thisBatchSize, getLastLayer()->getOutputPlanes(), &(labels[batchStart]), getResults() );
        StatefulTimer::timeCheck("after batch calc training num right");
        loss += calcLoss( &(expectedResults[batchStart*getOutputCubeSize()]) );
        StatefulTimer::timeCheck("after batch calc loss");
    }
    *p_totalCorrect = numRight;
    return loss;
}
//    float *propagate( int N, int batchSize, float const*images) {
//        float *results = new float[N];
//        int numBatches = N / batchSize;
//        for( int batch = 0; batch < numBatches; batch++ ) {
//            int batchStart = batch * batchSize;
//            int batchEndExcl = std::min( N, (batch + 1 ) * batchSize );
//            propagateBatch( &(images[batchStart]) );
//            std::cout << " batch " << batch << " start " << batchStart << " end " << batchEndExcl << std::endl;
//                float const *netResults = getResults();
//            for( int i = 0; i < batchSize; i++ ) {
//                results[batchStart + i ] = netResults[i];
//            }
//        }
//        return results;
//    }
void NeuralNet::propagate( float const*images) {
    // forward...
//        Timer timer;
    dynamic_cast<InputLayer *>(layers[0])->in( images );
    for( int layerId = 1; layerId < layers.size(); layerId++ ) {
        StatefulTimer::setPrefix("layer" + toString(layerId) + " " );
        layers[layerId]->propagate();
        StatefulTimer::setPrefix("" );
    }
//        timer.timeCheck("propagate time");
}
void NeuralNet::backPropFromLabels( float learningRate, int const *labels) {
    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if( acceptsLabels == 0 ) {
        throw std::runtime_error("Must add a child of IAcceptsLabels as last layer, to use backPropFromLabels");
    }
    acceptsLabels->calcErrorsFromLabels( labels );
    for( int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " " );
        Layer *layer = layers[layerIdx];
        if( layer->needsBackProp() ) {
            layer->backProp( learningRate );
        }
        StatefulTimer::setPrefix("" );
    }
}
void NeuralNet::backProp( float learningRate, float const *expectedResults) {
    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    if( lossLayer == 0 ) {
        throw std::runtime_error("Must add a LossLayer as last layer of net");
    }
    lossLayer->calcErrors( expectedResults );
    for( int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " " );
        layers[layerIdx]->backProp( learningRate );
        StatefulTimer::setPrefix("" );
    }
}
void NeuralNet::learnBatch( float learningRate, float const*images, float const *expectedResults ) {
    propagate( images);
    backProp( learningRate, expectedResults );
}
void NeuralNet::learnBatchFromLabels( float learningRate, float const*images, int const *labels ) {
    propagate( images);
    backPropFromLabels( learningRate, labels );
}
int NeuralNet::getNumLayers() {
    return (int)layers.size();
}
float const *NeuralNet::getResults( int layer ) const {
    return layers[layer]->getResults();
}
int NeuralNet::getInputCubeSize() const {
    return layers[ 0 ]->getOutputCubeSize();
}
int NeuralNet::getOutputCubeSize() const {
    return layers[ layers.size() - 1 ]->getOutputCubeSize();
}
float const *NeuralNet::getResults() const {
    return getResults( (int)layers.size() - 1 );
}
void NeuralNet::print() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        //(*it)->print();
        cout << (*it)->asString() << endl;
        i++;
    }
}
void NeuralNet::printWeights() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printWeights();
        i++;
    }
}
void NeuralNet::printOutput() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printOutput();
        i++;
    }
}

