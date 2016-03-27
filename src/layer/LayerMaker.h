// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>
#include <iostream>

#include "DeepCLDllExport.h"
#include "activate/ActivationFunction.h"

//#include "layer/Layer.h"
//#include "input/InputLayerMaker.h"
//#include "conv/ConvolutionalMaker.h"
//#include "activate/ActivationMaker.h"

class NeuralNet;
class Layer;
class EasyCL;

class SquareLossLayer;
class CrossEntropyLayer;
class SoftMaxLayer;

//class LayerMakerAny {
//    virtual void foo() { // just to maek it polymorphic...
//    }
//};

class DeepCL_EXPORT LayerMaker2 {
public:
    EasyCL *cl; // NOT owned by us
    LayerMaker2() :
        cl(0) {
    }
    virtual ~LayerMaker2() {}
    void setCl(EasyCL *cl) {
        this->cl = cl;
    }
    virtual Layer *createLayer(Layer *previousLayer) = 0;

    // see http://stackoverflow.com/questions/5148706/copying-a-polymorphic-object-in-c/5148751#5148751
    // apparently this is ok...
    virtual LayerMaker2 *clone() const = 0;
};

//class LayerMaker : public LayerMakerAny {
//public:
//    Layer *previousLayer;
//    NeuralNet *net; // only used for 'insert'
//    virtual int getOutputSize() const = 0;
//    virtual int getOutputPlanes() const = 0;
//    virtual int getBiased() const = 0;
//    virtual ActivationFunction const*getActivationFunction() const {
//        throw std::runtime_error("getactivationfunction not impelmented for this maker type");
//    }
//    LayerMaker(NeuralNet *net, Layer *previousLayer) :
//        net(net),
//        previousLayer(previousLayer) {
//    }
//    void setPreviousLayer(Layer *previousLayer) {
//        this->previousLayer = previousLayer;
//    }
//    virtual Layer *insert();
//    virtual Layer *instance() const = 0;
//    virtual LayerMaker *clone(Layer *clonePreviousLayer) const = 0;
//};

class DeepCL_EXPORT LossLayerMaker : public LayerMaker2 {
public:
//    Layer *previousLayer;
    LossLayerMaker() {
    }
    virtual LossLayerMaker *clone() const = 0;
};

class DeepCL_EXPORT SquareLossMaker : public LossLayerMaker {
public:
    PUBLICAPI SquareLossMaker() {
    }
    PUBLICAPI static SquareLossMaker *instance() {
        return new SquareLossMaker();
    }
    virtual SquareLossMaker *clone() const {
        return new SquareLossMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

class DeepCL_EXPORT CrossEntropyLossMaker : public LossLayerMaker {
public:
    PUBLICAPI CrossEntropyLossMaker() {
    }
    PUBLICAPI static CrossEntropyLossMaker *instance() {
        return new CrossEntropyLossMaker();
    }
    virtual CrossEntropyLossMaker *clone() const {
        return new CrossEntropyLossMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

// by default, it will be per-plane
// can switch to be per-column
class DeepCL_EXPORT SoftMaxMaker : public LossLayerMaker {
public:
    bool _perPlane; // = false;
    PUBLICAPI SoftMaxMaker() {
        _perPlane = false;
    }
    PUBLICAPI SoftMaxMaker *perColumn() {
        this->_perPlane = false;
        return clone();
    }
    PUBLICAPI SoftMaxMaker *perPlane() {
        this->_perPlane = true;
        return clone();
    }
    PUBLICAPI static SoftMaxMaker *instance() {
        return new SoftMaxMaker();
    }
    virtual SoftMaxMaker *clone() const {
        return new SoftMaxMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};


