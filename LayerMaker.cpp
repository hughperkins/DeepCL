// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "LayerMaker.h"

#include "NeuralNet.h"

#include <stdexcept>
using namespace std;

Layer *FullyConnectedMaker::insert() {
    if( _numPlanes == 0 ) {
        throw runtime_error("Must provide ->planes(planes)");
    }
    if( _boardSize == 0 ) {
        throw runtime_error("Must provide ->boardSize(boardSize)");
    }
//    Layer *layer = net->addFullyConnected( _numPlanes, _boardSize, _biased, _activationFunction );
    Layer *layer = net->addLayer( this );
    delete this;
    return layer;
}

Layer *InputLayerMaker::insert() {
    if( _numPlanes == 0 ) {
        throw runtime_error("Must provide ->planes(planes)");
    }
    if( _boardSize == 0 ) {
        throw runtime_error("Must provide ->boardSize(boardSize)");
    }
    Layer *layer = net->addLayer( this );
    delete this;
    return layer;
}

Layer *ConvolutionalMaker::insert() {
    if( _numFilters == 0 ) {
        throw runtime_error("Must provide ->numFilters(numFilters)");
    }
    if( _filterSize == 0 ) {
        throw runtime_error("Must provide ->filterSize(filterSize)");
    }
//    cout << _activationFunction.getKernelFunction();
//    Layer *previousLayer = net->layers[ net->layers.size() - 1 ];
//    Layer *layer = net->addLayer( new ConvolutionalLayer( previousLayer, this ) );
    std::cout << "insert()" << std::endl;
    Layer *layer = net->addLayer( this );
    std::cout << "insert() after net->addLayer" << std::endl;
 //   layer->print();
    std::cout << "insert() after layer->print" << std::endl;
//    Layer *layer = net->addConvolutional( _numFilters, _filterSize, _padZeros, _biased, _activationFunction );
    delete this;
    return layer;
}

Layer *FullyConnectedMaker::instance() const {
    Layer *layer = new FullyConnectedLayer( previousLayer, this );
    return layer;
}

Layer *ConvolutionalMaker::instance() const {
    std::cout << "ConvolutionalMaker::instance()" << std::endl;
    Layer *layer = new ConvolutionalLayer( previousLayer, this );
    return layer;
}

Layer *InputLayerMaker::instance() const {
    Layer *layer = new InputLayer( 0, this );
    return layer;
}

int ConvolutionalMaker::getBoardSize() const {
    std::cout << "getBoardSize" << std::endl;
    if( previousLayer == 0 ) {
        throw std::runtime_error("convolutional network must be attached to a parent layer");
    }
    std::cout << "getBoardSize 2" << std::endl;
    int boardSize = _padZeros ? previousLayer->boardSize : previousLayer->boardSize - _filterSize + 1;
    std::cout << "getBoardSize 3 result " << boardSize << std::endl;
    return boardSize;
}

