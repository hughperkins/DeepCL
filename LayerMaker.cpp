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
    Layer *layer = net->addLayer( this );
    delete this;
    return layer;
}

Layer *FullyConnectedMaker::instance() const {
    Layer *layer = new FullyConnectedLayer( previousLayer, this );
    return layer;
}

Layer *ConvolutionalMaker::instance() const {
    Layer *layer = new ConvolutionalLayer( previousLayer, this );
    return layer;
}

Layer *InputLayerMaker::instance() const {
    Layer *layer = new InputLayer( 0, this );
    return layer;
}

int ConvolutionalMaker::getBoardSize() const {
    if( previousLayer == 0 ) {
        throw std::runtime_error("convolutional network must be attached to a parent layer");
    }
    int boardSize = _padZeros ? previousLayer->boardSize : previousLayer->boardSize - _filterSize + 1;
    return boardSize;
}

