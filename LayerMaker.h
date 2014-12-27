// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

//#include "NeuralNet.h"
#include "Layer.h"

class NeuralNet;

class FullyConnectedMaker {
    NeuralNet *net;
    int _numPlanes;
    int _boardSize;
public:
    FullyConnectedMaker( NeuralNet *net ) :
        net(net) {
        std::cout << "FullyConnectedMaker()" << std::endl;
    }
    FullyConnectedMaker *planes(int numPlanes) {
        this->_numPlanes = numPlanes;
        return this;
    }    
    FullyConnectedMaker *boardSize(int boardSize) {
        this->_boardSize = boardSize;
        return this;
    }
    Layer *make();
};

class ConvolutionalMaker {
    NeuralNet *net;
    int _numFilters;
    int _filterSize;
public:
    ConvolutionalMaker( NeuralNet *net ) 
         : net(net) {
        std::cout << "ConvolutionalMaker()" << std::endl;
    }
    ConvolutionalMaker *filters(int numFilters) {
        this->_numFilters = numFilters;
        return this;
    }    
    ConvolutionalMaker *filterSize(int filterSize) {
        this->_filterSize = filterSize;
        return this;
    }    
    Layer *make();
};

