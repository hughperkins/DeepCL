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
//#include "FullyConnectedLayer.h"
#include "EpochMaker.h"
//#include "ExpectedValuesLayer.h"
#include "LossLayer.h"
#include "ExceptionMacros.h"

#include "NeuralNet.h"

using namespace std;

//static std::mt19937 random;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

NeuralNet::NeuralNet( int numPlanes, int boardSize ) {
//    cout << "NeuralNet() begin" << endl;
    cl = new OpenCLHelper();
    InputLayerMaker *maker = new InputLayerMaker( this, numPlanes, boardSize );
    maker->insert();

//    ExpectedValuesLayer *expectedValuesLayer = ( new ExpectedValuesLayerMaker( this, getLastLayer() ) )->instance();
//    expectedValuesLayer->setBatchSize(batchSize);
//    getLastLayer()->nextLayer = expectedValuesLayer;
//    layers.push_back( expectedValuesLayer );
//    cout << "NeuralNet() end" << endl;
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
//FullyConnectedMaker *NeuralNet::fullyConnectedMaker() {
//    return new FullyConnectedMaker( this );
//}
ConvolutionalMaker *NeuralNet::convolutionalMaker() {
    return new ConvolutionalMaker( this );
}
SquareLossMaker *NeuralNet::squareLossMaker() {
    return new SquareLossMaker( this, getLastLayer() );
}
CrossEntropyLossMaker *NeuralNet::crossEntropyLossMaker() {
    return new CrossEntropyLossMaker( this, getLastLayer() );
}
SoftMaxMaker *NeuralNet::softMaxMaker() {
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
//    cout << "NeuralNet::addLayer() begin" << endl;
//    // first, remove the expectedvalueslayer
//    if( layers.size() > 1 ) {
//        delete layers[ layers.size() - 1 ];
//        layers.erase( layers.end() - 1 );
//    }

    // then add the new layer
    Layer *previousLayer = 0;
    if( layers.size() > 0 ) {
        previousLayer = layers[ layers.size() - 1 ];
    }
    maker->setPreviousLayer( previousLayer );
    Layer *layer = maker->instance();
    layers.push_back( layer );

//    if( layers.size() > 1 ) {
//        ExpectedValuesLayer *expectedValuesLayer = ( new ExpectedValuesLayerMaker( this, getLastLayer() ) )->instance();
//        getLastLayer()->nextLayer = expectedValuesLayer;
//        layers.push_back( expectedValuesLayer );
//    }

//    cout << "NeuralNet::addLayer() end" << endl;
    // then put back on an expectedvalues layer
    return layer;
}
void NeuralNet::setBatchSize( int batchSize ) {
    for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
        (*it)->setBatchSize( batchSize );
    }
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
//        StatefulTimer::dump();
//        timer.timeCheck("epoch time");
//    layers.erase( layers.end() - 1 );
//    getLastLayer()->nextLayer = 0;
//    delete expectedValuesLayer;
    return loss;
}
float NeuralNet::doEpochWithCalcTrainingAccuracy( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults, int const *labels, int *p_totalCorrect ) {
//        Timer timer;
//    ExpectedValuesLayer *expectedValuesLayer = ( new ExpectedValuesLayerMaker( this, getLastLayer() ) )->instance();
////    expectedValuesLayer->setBatchSize(batchSize);
//    getLastLayer()->nextLayer = expectedValuesLayer;
//    layers.push_back( expectedValuesLayer );
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
//        StatefulTimer::dump();
//        timer.timeCheck("epoch time");
//    layers.erase( layers.end() - 1 );
//    getLastLayer()->nextLayer = 0;
//    delete expectedValuesLayer;
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
void NeuralNet::backProp( float learningRate, float const *expectedResults) {
    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    if( lossLayer == 0 ) {
        throw std::runtime_error("Must add a LossLayer as last layer of net");
    }
    lossLayer->calcErrors( expectedResults );
    for( int layerIdx = layers.size() - 2; layerIdx >= 1; layerIdx-- ) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " " );
        layers[layerIdx]->backProp( learningRate );
        StatefulTimer::setPrefix("" );
    }
}
void NeuralNet::learnBatch( float learningRate, float const*images, float const *expectedResults ) {
//        Timer timer;
    propagate( images);
//        timer.timeCheck("propagate");
    backProp( learningRate, expectedResults );
//        timer.timeCheck("backProp");
}
int NeuralNet::getNumLayers() {
    return layers.size();
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

