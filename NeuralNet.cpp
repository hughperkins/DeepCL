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
#include "ExpectedValuesLayer.h"

#include "NeuralNet.h"

using namespace std;

//static std::mt19937 random;

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
// [static]
NeuralNetMould *NeuralNet::maker() {
    return new NeuralNetMould();
}
FullyConnectedMaker *NeuralNet::fullyConnectedMaker() {
    return new FullyConnectedMaker( this );
}
ConvolutionalMaker *NeuralNet::convolutionalMaker() {
    return new ConvolutionalMaker( this );
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
    return layers[layers.size()-1]->calcLoss( expectedValues );
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
    maker->setPreviousLayer( previousLayer );
    Layer *layer = maker->instance();
    layers.push_back( layer );
    return layer;
}
void NeuralNet::setBatchSize( int batchSize ) {
    for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        (*it)->setBatchSize( batchSize );
    }
}
float NeuralNet::doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults ) {
//        Timer timer;
    setBatchSize( batchSize );
    int numBatches = numImages / batchSize;
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
        learnBatch( batchSize, learningRate, &(images[batchStart*getInputSizePerExample()]), &(expectedResults[batchStart*getResultsSizePerExample()]) );
        loss += calcLoss( &(expectedResults[batchStart*getResultsSizePerExample()]) );
    }
//        StatefulTimer::dump();
//        timer.timeCheck("epoch time");
    return loss;
}
float NeuralNet::doEpochWithCalcTrainingAccuracy( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults, int const *labels, int *p_totalCorrect ) {
//        Timer timer;
    setBatchSize( batchSize );
    int numBatches = ( numImages + batchSize - 1 ) / batchSize;
    std::cout << "numBatches: " << numBatches << std::endl;
    float loss = 0;
    int numRight = 0;
    int total = 0;
    if( getLastLayer()->boardSize != 1 ) {
        throw std::runtime_error("Last layer should have board size of 1, and number of planes equal number of categories, if you want to measure training accuracy");
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
        learnBatch( batchSize, learningRate, &(images[batchStart*getInputSizePerExample()]), &(expectedResults[batchStart*getResultsSizePerExample()]) );
        StatefulTimer::timeCheck("after batch forward-backward prop");
        numRight += AccuracyHelper::calcNumRight( thisBatchSize, getLastLayer()->numPlanes, &(labels[batchStart]), getResults() );
        StatefulTimer::timeCheck("after batch calc training num right");
        loss += calcLoss( &(expectedResults[batchStart*getResultsSizePerExample()]) );
        StatefulTimer::timeCheck("after batch calc loss");
    }
    *p_totalCorrect = numRight;
//        StatefulTimer::dump();
//        timer.timeCheck("epoch time");
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
        layers[layerId]->propagate();
    }
//        timer.timeCheck("propagate time");
}
void NeuralNet::backProp( int batchSize, float learningRate, float const *expectedResults) {
    // backward...
    ExpectedValuesLayer *expectedValuesLayer = ( new ExpectedValuesLayerMaker( this, getLastLayer() ) )->instance();
    expectedValuesLayer->setBatchSize(batchSize);
//    Layer *lastLayer = getLastLayer();
//    float *errors = new float[ lastLayer->getResultsSize() ];
    expectedValuesLayer->calcErrors( expectedResults );
//    float *errorsForNextLayer = 0;
    for( int layerIdx = layers.size() - 1; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
//        if( layerIdx > 1 ) {
//            errorsForNextLayer = new float[ layers[layerIdx-1]->getResultsSize() ];
//        }
        if( layerIdx == layers.size() - 1 ) {
            layers[layerIdx]->backPropErrors( learningRate, expectedValuesLayer );
        } else {
            layers[layerIdx]->backPropErrors( learningRate, layers[layerIdx+1] );
        }
//        delete[] errors;
//        errors = 0;
//        errors = errorsForNextLayer;
//        errorsForNextLayer = 0;
    }
    delete expectedValuesLayer;
}
void NeuralNet::learnBatch( int batchSize, float learningRate, float const*images, float const *expectedResults ) {
//        Timer timer;
    propagate( images);
//        timer.timeCheck("propagate");
    backProp(batchSize, learningRate, expectedResults );
//        timer.timeCheck("backProp");
}
int NeuralNet::getNumLayers() {
    return layers.size();
}
float const *NeuralNet::getResults( int layer ) const {
    return layers[layer]->getResults();
}
int NeuralNet::getInputSizePerExample() const {
    return layers[ 0 ]->getResultsSizePerExample();
}
int NeuralNet::getResultsSizePerExample() const {
    return layers[ layers.size() - 1 ]->getResultsSizePerExample();
}
float const *NeuralNet::getResults() const {
    return getResults( layers.size() - 1 );
}
void NeuralNet::print() {
    int i = 0; 
    for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
        std::cout << "layer " << i << ":" << std::endl;
        (*it)->print();
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

