// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

//#include "NeuralNet.h"
#include "Layer.h"
#include "ActivationFunction.h"

class NeuralNet;

class FullyConnectedMaker {
    NeuralNet *net;
    int _numPlanes;
    int _boardSize;
    int _biased;
    ActivationFunction *_activationFunction;
public:
    FullyConnectedMaker( NeuralNet *net ) :
        net(net) {
        _activationFunction = new TanhActivation();
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
    FullyConnectedMaker *biased() {
        this->_biased = true;
        return this;
    }    
    FullyConnectedMaker *biased(int _biased) {
        this->_biased = _biased;
        return this;
    }    
    FullyConnectedMaker *linear() {
        delete this->_activationFunction;
        this->_activationFunction = new LinearActivation();
        return this;
    }
    FullyConnectedMaker *tanh() {
        delete this->_activationFunction;
        this->_activationFunction = new TanhActivation();
        return this;
    }
    FullyConnectedMaker *relu() {
        delete this->_activationFunction;
        this->_activationFunction = new ReluActivation();
        return this;
    }
    Layer *insert();
};

class ConvolutionalMaker {
    NeuralNet *net;
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    bool _biased;
    ActivationFunction *_activationFunction;
public:
    ConvolutionalMaker( NeuralNet *net ) {
        memset( this, 0, sizeof( ConvolutionalMaker ) );
        this->net = net;
        _activationFunction = new TanhActivation();
        std::cout << "ConvolutionalMaker()" << std::endl;
    }
    ConvolutionalMaker *numFilters(int numFilters) {
        this->_numFilters = numFilters;
        return this;
    }    
    ConvolutionalMaker *filterSize(int filterSize) {
        this->_filterSize = filterSize;
        return this;
    }    
    ConvolutionalMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }    
    ConvolutionalMaker *biased() {
        this->_biased = true;
        return this;
    }    
    ConvolutionalMaker *biased(int _biased) {
        this->_biased = _biased;
        return this;
    }    
    ConvolutionalMaker *tanh() {
        delete this->_activationFunction;
        this->_activationFunction = new TanhActivation();
        return this;
    }
    ConvolutionalMaker *relu() {
        delete this->_activationFunction;
        this->_activationFunction = new ReluActivation();
        return this;
    }
    ConvolutionalMaker *linear() {
        delete this->_activationFunction;
        this->_activationFunction = new LinearActivation();
        return this;
    }
    Layer *insert();
};

