// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <vector>
#include <random>

#include "Timer.h"
#include "Layer.h"
#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "NeuralNetMould.h"
#include "LayerMaker.h"
#include "EpochMaker.h"
#include "ActivationFunction.h"

class NeuralNet {
public:
    std::vector< Layer *> layers;
    NeuralNet( int numPlanes, int boardSize ) {
        InputLayerMaker *maker = new InputLayerMaker( this, numPlanes, boardSize );
        maker->insert();
//        InputLayer *inputLayer = new InputLayer( numPlanes, boardSize );
//        layers.push_back( inputLayer );
    }
    ~NeuralNet() {
        for( int i = 0; i < layers.size(); i++ ) {
            delete layers[i];
        }
    }
    static NeuralNetMould *maker() {
//        NeuralNetMould neuralNetMould;
        return new NeuralNetMould();
    }
    FullyConnectedMaker *fullyConnectedMaker() {
        return new FullyConnectedMaker( this );
    }
    ConvolutionalMaker *convolutionalMaker() {
        return new ConvolutionalMaker( this );
    }
    void initWeights( int layerIndex, float *weights, float *biasWeights ) {
        initWeights( layerIndex, weights );
        initBiasWeights( layerIndex, biasWeights );
    }
    void initWeights( int layerIndex, float *weights ) {
        layers[layerIndex]->initWeights( weights );
    }
    void initBiasWeights( int layerIndex, float *weights ) {
        layers[layerIndex]->initBiasWeights( weights );
    }
    void printWeightsAsCode() {
        for( int layer = 1; layer < layers.size(); layer++ ) {
            layers[layer]->printWeightsAsCode();
        }
    }
    void printBiasWeightsAsCode() {
        for( int layer = 1; layer < layers.size(); layer++ ) {
            layers[layer]->printBiasWeightsAsCode();
        }
    }
    float calcLoss(float const *expectedValues ) {
        return layers[layers.size()-1]->calcLoss( expectedValues );
    }
//    Layer *insertLayer( Layer *layer ) {
//        if( layers.size() > 0 ) {
//            layer->parent = layers[layers.size() - 1];
//        } 
//        layers.push_back(layer);
//        return layer;
//    }
    EpochMaker *epochMaker() {
         return new EpochMaker(this);
    }
    Layer *addLayer( LayerMaker *maker ) {
        Layer *previousLayer = 0;
        if( layers.size() > 0 ) {
            previousLayer = layers[ layers.size() - 1 ];
        }
        std::cout << "1" << std::endl;
        maker->setPreviousLayer( previousLayer );
        std::cout << "1" << std::endl;
        Layer *layer = maker->instance();
        std::cout << "1" << std::endl;
        layers.push_back( layer );
        std::cout << "1" << std::endl;
        return layer;
    }
//    Layer *addFullyConnected( int numOutputPlanes, int outputBoardSize, bool biased, ActivationFunction *fn ) {
//        Layer *layer = new FullyConnectedLayer(layers[layers.size() - 1], numOutputPlanes, outputBoardSize, biased, fn );
//        layers.push_back( layer );
//        return layer;
//    }
//    Layer *addConvolutional( int numFilters, int filterSize, bool padZeros, bool biased, ActivationFunction *fn ) {
//        Layer *layer = new ConvolutionalLayer(layers[layers.size() - 1], numFilters, filterSize, padZeros, biased, fn );
//        layers.push_back( layer );
//        return layer;
//    }
    void setBatchSize( int batchSize ) {
        for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
            (*it)->setBatchSize( batchSize );
        }
    }
    float doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults ) {
        Timer timer;
        setBatchSize( batchSize );
        int numBatches = numImages / batchSize;
        float loss = 0;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * batchSize;
//            std::cout << " batch " << batch << " start " << batchStart << " inputsizeperex " << getInputSizePerExample() <<
//             " resultssizeperex " << getResultsSizePerExample() << std::endl;
            learnBatch( learningRate, &(images[batchStart*getInputSizePerExample()]), &(expectedResults[batchStart*getResultsSizePerExample()]) );
            loss += calcLoss( &(expectedResults[batchStart*getResultsSizePerExample()]) );
        }
        timer.timeCheck("epoch time");
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
    void propagate( float const*images) {
        // forward...
//        Timer timer;
        dynamic_cast<InputLayer *>(layers[0])->in( images );
        for( int layerId = 1; layerId < layers.size(); layerId++ ) {
            layers[layerId]->propagate();
        }
//        timer.timeCheck("propagate time");
    }
    void backProp(float learningRate,float const *expectedResults) {
        // backward...
        layers[layers.size() - 1]->backPropExpected( learningRate, expectedResults );
    }
    void learnBatch( float learningRate, float const*images, float const *expectedResults ) {
        Timer timer;
        propagate( images);
        timer.timeCheck("propagate");
        backProp(learningRate, expectedResults );
        timer.timeCheck("backProp");
    }
    int getNumLayers() {
        return layers.size();
    }
    float const *getResults( int layer ) const {
        return layers[layer]->getResults();
    }
    int getInputSizePerExample() const {
        return layers[ 0 ]->getResultsSizePerExample();
    }
    int getResultsSizePerExample() const {
        return layers[ layers.size() - 1 ]->getResultsSizePerExample();
    }
    float const *getResults() const {
        return getResults( layers.size() - 1 );
    }
    void print() {
        int i = 0; 
        for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
            std::cout << "layer " << i << ":" << std::endl;
            (*it)->print();
            i++;
        }
    }
    void printWeights() {
        int i = 0; 
        for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
            std::cout << "layer " << i << ":" << std::endl;
            (*it)->printWeights();
            i++;
        }
    }
    void printOutput() {
        int i = 0; 
        for( std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++ ) {
            std::cout << "layer " << i << ":" << std::endl;
            (*it)->printOutput();
            i++;
        }
    }
};



