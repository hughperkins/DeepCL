// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>
#include <iostream>

#include "ActivationFunction.h"

class NeuralNet;
class Layer;
class OpenCLHelper;

class SquareLossLayer;
class CrossEntropyLayer;
class SoftMaxLayer;

class LayerMakerAny {
    virtual void foo() { // just to maek it polymorphic...
    }
};

class LayerMaker2 : public LayerMakerAny {
public:
    OpenCLHelper *cl; // NOT owned by us
    LayerMaker2() :
        cl(0) {
    }
    void setCl( OpenCLHelper *cl ) {
        this->cl = cl;
    }
    virtual Layer *createLayer( Layer *previousLayer ) = 0;
};

class LayerMaker : public LayerMakerAny {
public:
    Layer *previousLayer;
    NeuralNet *net; // only used for 'insert'
    virtual int getOutputBoardSize() const = 0;
    virtual int getOutputPlanes() const = 0;
    virtual int getBiased() const = 0;
    virtual ActivationFunction const*getActivationFunction() const {
        throw std::runtime_error("getactivationfunction not impelmented for this maker type");
    }
    LayerMaker( NeuralNet *net, Layer *previousLayer ) :
        net( net ),
        previousLayer( previousLayer ) {
    }
    void setPreviousLayer( Layer *previousLayer ) {
        this->previousLayer = previousLayer;
    }
    virtual Layer *insert();
    virtual Layer *instance() const = 0;
    virtual LayerMaker *clone( Layer *clonePreviousLayer ) const = 0;
};

class NormalizationLayerMaker : public LayerMaker {
public:
    float _translate;
    float _scale;
    NormalizationLayerMaker( NeuralNet *net, Layer *previousLayer ) :
        LayerMaker( net, previousLayer ),
        _translate(0.0f),
        _scale( 1.0f ) {
    }
    NormalizationLayerMaker *translate( float _translate ) {
        this->_translate = _translate;
        return this;
    }
    NormalizationLayerMaker *scale( float _scale ) {
        this->_scale = _scale;
        return this;
    }
    virtual int getOutputBoardSize() const;
    virtual int getOutputPlanes() const;
    virtual int getBiased() const;
    virtual Layer *instance() const;
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};

class PoolingMaker : public LayerMaker {
public:
//    Layer *previousLayer;
    int _poolingSize = 2;
    bool _padZeros = false;
    PoolingMaker( NeuralNet *net, Layer *previousLayer ) :
        LayerMaker( net, previousLayer ) {
    }
    PoolingMaker *poolingSize( int _poolingSize ) {
        this->_poolingSize = _poolingSize;
        return this;
    }
    PoolingMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }
    virtual int getOutputBoardSize() const;
    virtual int getOutputPlanes() const;
    virtual int getBiased() const;
    virtual Layer *instance() const;
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};

class LossLayerMaker : public LayerMaker {
public:
//    Layer *previousLayer;
    LossLayerMaker( NeuralNet *net, Layer *previousLayer ) :
        LayerMaker( net, previousLayer ) {
    }
    virtual int getOutputBoardSize() const;
    virtual int getOutputPlanes() const;
    virtual int getBiased() const;
};

class SquareLossMaker : public LossLayerMaker {
public:
    SquareLossMaker( NeuralNet *net, Layer *previousLayer ) :
        LossLayerMaker( net, previousLayer ) {
    }
    virtual Layer *instance() const;
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};

class CrossEntropyLossMaker : public LossLayerMaker {
public:
    CrossEntropyLossMaker( NeuralNet *net, Layer *previousLayer ) :
        LossLayerMaker( net, previousLayer ) {
    }
    virtual Layer *instance() const;
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};

// by default, it will be per-plane
// can switch to be per-column
class SoftMaxMaker : public LossLayerMaker {
public:
    bool _perPlane = false;
    SoftMaxMaker( NeuralNet *net, Layer *previousLayer ) :
        LossLayerMaker( net, previousLayer ) {
    }
    virtual Layer *instance() const;
    SoftMaxMaker *perColumn() {
        this->_perPlane = false;
        return this;
    }
    SoftMaxMaker *perPlane() {
        this->_perPlane = true;
        return this;
    }
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};

class FullyConnectedMaker : public LayerMaker {
public:
    int _numPlanes;
    int _boardSize;
    int _biased;
    ActivationFunction *_activationFunction;
    FullyConnectedMaker( NeuralNet *net, Layer *previousLayer ) :
        LayerMaker(net, previousLayer),
        _numPlanes(0),
        _boardSize(0),
        _activationFunction( new TanhActivation() ) {
    }
    virtual int getOutputBoardSize() const {
        return _boardSize;
    }
    virtual int getOutputPlanes() const {
        return _numPlanes;
    }
    virtual int getBiased() const {
        return _biased;
    }
    virtual ActivationFunction const *getActivationFunction() const {
        return _activationFunction;
    }
    FullyConnectedMaker *numPlanes(int numPlanes) {
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
    FullyConnectedMaker *sigmoid() {
        delete this->_activationFunction;
        this->_activationFunction = new SigmoidActivation();
        return this;
    }
    FullyConnectedMaker *relu() {
        delete this->_activationFunction;
        this->_activationFunction = new ReluActivation();
        return this;
    }
//    virtual Layer *insert();
    virtual Layer *instance() const;
    virtual LayerMaker *clone( Layer *previousLayer ) const;
};


