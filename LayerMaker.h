// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>
#include <iostream>

//#include "NeuralNet.h"
//#include "Layer.h"
#include "ActivationFunction.h"

class NeuralNet;
class Layer;

//class ExpectedValuesLayer;
class SquareLossLayer;
class CrossEntropyLayer;

class LayerMaker {
public:
    Layer *previousLayer;
    NeuralNet *net;
    virtual int getBoardSize() const = 0;
    virtual int getNumPlanes() const = 0;
    virtual int getBiased() const = 0;
    virtual ActivationFunction const*getActivationFunction() const {
        throw std::runtime_error("getactivationfunction not impelmented for this maker type");
    }
    LayerMaker( NeuralNet *net ) :
        net( net ) {
    }
    void setPreviousLayer( Layer *previousLayer ) {
        this->previousLayer = previousLayer;
    }
    virtual Layer *insert() = 0;
    virtual Layer *instance() const = 0;
};

class InputLayerMaker : public LayerMaker {
public:
    int _numPlanes;
    int _boardSize;
    InputLayerMaker( NeuralNet *net, int numPlanes, int boardSize ) :
            LayerMaker( net ),
            _numPlanes(numPlanes),
            _boardSize(boardSize) {
    }
    virtual int getBoardSize() const {
        return _boardSize;
    }
    virtual int getNumPlanes() const {
        return _numPlanes;
    }
    virtual int getBiased() const {
        return false;
    }
    virtual Layer *instance() const;
    virtual Layer *insert();
};

//class ExpectedValuesLayerMaker {
//public:
//    NeuralNet *net;
//    Layer *previousLayer;
//    ExpectedValuesLayerMaker( NeuralNet *net, Layer *previousLayer ) :
//        net( net ),
//        previousLayer( previousLayer ) {
//    }
//    virtual ExpectedValuesLayer *instance() const;
////    virtual Layer *insert();
//};

//class SoftMaxMaker : public LayerMaker {
//public:
//    SoftMaxMaker( NeuralNet *net ) :
//        LayerMaker( net ) {
//    }
//    virtual Layer *insert();
//    virtual Layer *instance() const;
//};

class LossLayerMaker : public LayerMaker {
public:
    Layer *previousLayer;
    LossLayerMaker( NeuralNet *net, Layer *previousLayer ) :
        LayerMaker( net ),
        previousLayer( previousLayer ) {
    }
    virtual int getBoardSize() const;
    virtual int getNumPlanes() const;
    virtual int getBiased() const;
    virtual Layer *insert();
};

class SquareLossMaker : public LossLayerMaker {
public:
    SquareLossMaker( NeuralNet *net, Layer *previousLayer ) :
        LossLayerMaker( net, previousLayer ) {
    }
    virtual Layer *instance() const;
};

class CrossEntropyLossMaker : public LossLayerMaker {
public:
    CrossEntropyLossMaker( NeuralNet *net, Layer *previousLayer ) :
        LossLayerMaker( net, previousLayer ) {
    }
    virtual Layer *instance() const;
};

//class FullyConnectedMaker : public LayerMaker {
//public:
//    int _numPlanes;
//    int _boardSize;
//    int _biased;
//    ActivationFunction *_activationFunction;
//    FullyConnectedMaker( NeuralNet *net ) :
//        LayerMaker(net),
//        _numPlanes(0),
//        _boardSize(0),
//        _activationFunction( new TanhActivation() ) {
//    }
//    virtual int getBoardSize() const {
//        return _boardSize;
//    }
//    virtual int getNumPlanes() const {
//        return _numPlanes;
//    }
//    virtual int getBiased() const {
//        return _biased;
//    }
//    virtual ActivationFunction const *getActivationFunction() const {
//        return _activationFunction;
//    }
//    FullyConnectedMaker *planes(int numPlanes) {
//        this->_numPlanes = numPlanes;
//        return this;
//    }    
//    FullyConnectedMaker *boardSize(int boardSize) {
//        this->_boardSize = boardSize;
//        return this;
//    }
//    FullyConnectedMaker *biased() {
//        this->_biased = true;
//        return this;
//    }    
//    FullyConnectedMaker *biased(int _biased) {
//        this->_biased = _biased;
//        return this;
//    }    
//    FullyConnectedMaker *linear() {
//        delete this->_activationFunction;
//        this->_activationFunction = new LinearActivation();
//        return this;
//    }
//    FullyConnectedMaker *tanh() {
//        delete this->_activationFunction;
//        this->_activationFunction = new TanhActivation();
//        return this;
//    }
//    FullyConnectedMaker *relu() {
//        delete this->_activationFunction;
//        this->_activationFunction = new ReluActivation();
//        return this;
//    }
//    virtual Layer *insert();
//    virtual Layer *instance() const;
//};

class ConvolutionalMaker : public LayerMaker {
public:
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    int _biased;
    ActivationFunction *_activationFunction;
    ConvolutionalMaker( NeuralNet *net ) :
            LayerMaker( net ),
            _numFilters(0),
            _filterSize(0),
            _padZeros(false),
        _activationFunction( new TanhActivation() ) {
    }
    virtual int getBoardSize() const;
    virtual int getNumPlanes() const {
        return _numFilters;
    }
    virtual int getBiased() const {
        return _biased;
    }
    virtual ActivationFunction const*getActivationFunction() const {
        return _activationFunction;
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
    ConvolutionalMaker *padZeros( bool value ) {
        this->_padZeros = value;
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
    ConvolutionalMaker *sigmoid() {
        delete this->_activationFunction;
        this->_activationFunction = new SigmoidActivation();
        return this;
    }
    ConvolutionalMaker *linear() {
        delete this->_activationFunction;
        this->_activationFunction = new LinearActivation();
        return this;
    }
    ConvolutionalMaker *fn(ActivationFunction *_fn) {
        this->_activationFunction = _fn;
        return this;
    }
    virtual Layer *insert();
    virtual Layer *instance() const;
};

