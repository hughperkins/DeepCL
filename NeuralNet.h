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

class NeuralNet {
public:
    std::vector< Layer *> layers;
    NeuralNet( int numPlanes, int boardSize ) {
        InputLayer *inputLayer = new InputLayer( numPlanes, boardSize );
        layers.push_back( inputLayer );
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
    Layer *addFullyConnected( int numOutputPlanes, int outputBoardSize ) {
        Layer *layer = new FullyConnectedLayer(layers[layers.size() - 1], numOutputPlanes, outputBoardSize );
        layers.push_back( layer );
        return layer;
    }
    Layer *addConvolutional( int numFilters, int filterSize, bool padZeros = true, bool biased = true ) {
        Layer *layer = new ConvolutionalLayer(layers[layers.size() - 1], numFilters, filterSize, padZeros, biased );
        layers.push_back( layer );
        return layer;
    }
    void setBatchSize( int batchSize ) {
        for( std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++ ) {
            (*it)->setBatchSize( batchSize );
        }
    }
    void doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults ) {
        setBatchSize( batchSize );
        int numBatches = numImages / batchSize;
        std::cout << "numbatches " << numBatches << " batchsize " << batchSize << std::endl;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * batchSize;
            int batchEndExcl = std::min( numImages, (batch + 1 ) * batchSize );
            std::cout << " batch " << batch << " start " << batchStart << " end " << batchEndExcl << std::endl;
//            learnBatch( learningRate, batchStart, batchEndExcl, images, expectedResults );
//            learnBatch( learningRate, &(images[batchStart]), &(expectedResults[batchStart]) );
            learnBatch( learningRate, images, expectedResults );
        }
    }
    float *propagate( int N, int batchSize, float const*images) {
        float *results = new float[N];
        int numBatches = N / batchSize;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * batchSize;
            int batchEndExcl = std::min( N, (batch + 1 ) * batchSize );
            propagateBatch( &(images[batchStart]) );
            std::cout << " batch " << batch << " start " << batchStart << " end " << batchEndExcl << std::endl;
                float const *netResults = getResults();
            for( int i = 0; i < batchSize; i++ ) {
                results[batchStart + i ] = netResults[i];
            }
        }
        return results;
    }
    void propagateBatch( float const*images) {
        // forward...
        dynamic_cast<InputLayer *>(layers[0])->in( images );
        for( int layerId = 1; layerId < layers.size(); layerId++ ) {
            layers[layerId]->propagate();
        }
    }
    void backProp(float learningRate,float const *expectedResults) {
        // backward...
        layers[layers.size() - 1]->backPropExpected( learningRate, expectedResults );
    }
    void learnBatch( float learningRate, float const*images, float const *expectedResults ) {
        Timer timer;
        propagateBatch( images);
//        timer.timeCheck("propagate");
        backProp(learningRate, expectedResults );
//        timer.timeCheck("backProp");
    }
    int getNumLayers() {
        return layers.size();
    }
    float const *getResults( int layer ) const {
        return layers[layer]->getResults();
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
};



