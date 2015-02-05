// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
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


