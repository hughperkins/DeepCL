// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>
#include <iostream>

#include "ClConvolveDllExport.h"
#include "ActivationFunction.h"

class NeuralNet;
class Layer;
class OpenCLHelper;

class SquareLossLayer;
class CrossEntropyLayer;
class SoftMaxLayer;

//class LayerMakerAny {
//    virtual void foo() { // just to maek it polymorphic...
//    }
//};

class ClConvolve_EXPORT LayerMaker2 {
public:
    OpenCLHelper *cl; // NOT owned by us
    LayerMaker2() :
        cl(0) {
    }
    void setCl( OpenCLHelper *cl ) {
        this->cl = cl;
    }
    virtual Layer *createLayer( Layer *previousLayer ) = 0;
    virtual LayerMaker2 *clone() const = 0;
};

//class LayerMaker : public LayerMakerAny {
//public:
//    Layer *previousLayer;
//    NeuralNet *net; // only used for 'insert'
//    virtual int getOutputBoardSize() const = 0;
//    virtual int getOutputPlanes() const = 0;
//    virtual int getBiased() const = 0;
//    virtual ActivationFunction const*getActivationFunction() const {
//        throw std::runtime_error("getactivationfunction not impelmented for this maker type");
//    }
//    LayerMaker( NeuralNet *net, Layer *previousLayer ) :
//        net( net ),
//        previousLayer( previousLayer ) {
//    }
//    void setPreviousLayer( Layer *previousLayer ) {
//        this->previousLayer = previousLayer;
//    }
//    virtual Layer *insert();
//    virtual Layer *instance() const = 0;
//    virtual LayerMaker *clone( Layer *clonePreviousLayer ) const = 0;
//};

class ClConvolve_EXPORT LossLayerMaker : public LayerMaker2 {
public:
//    Layer *previousLayer;
    LossLayerMaker() {
    }
};

class ClConvolve_EXPORT SquareLossMaker : public LossLayerMaker {
public:
    SquareLossMaker() {
    }
    static SquareLossMaker *instance() {
        return new SquareLossMaker();
    }
    virtual SquareLossMaker *clone() const {
        SquareLossMaker *thisClone = new SquareLossMaker();
        memcpy( thisClone, this, sizeof( SquareLossMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

class ClConvolve_EXPORT CrossEntropyLossMaker : public LossLayerMaker {
public:
    CrossEntropyLossMaker() {
    }
    static CrossEntropyLossMaker *instance() {
        return new CrossEntropyLossMaker();
    }
    virtual CrossEntropyLossMaker *clone() const {
        CrossEntropyLossMaker *thisClone = new CrossEntropyLossMaker();
        memcpy( thisClone, this, sizeof( CrossEntropyLossMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

// by default, it will be per-plane
// can switch to be per-column
class ClConvolve_EXPORT SoftMaxMaker : public LossLayerMaker {
public:
    bool _perPlane = false;
    SoftMaxMaker() {
    }
    SoftMaxMaker *perColumn() {
        this->_perPlane = false;
        return this;
    }
    SoftMaxMaker *perPlane() {
        this->_perPlane = true;
        return this;
    }
    static SoftMaxMaker *instance() {
        return new SoftMaxMaker();
    }
    virtual SoftMaxMaker *clone() const {
        SoftMaxMaker *thisClone = new SoftMaxMaker();
        memcpy( thisClone, this, sizeof( SoftMaxMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};


