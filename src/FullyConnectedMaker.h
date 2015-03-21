// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "LayerMaker.h"
#include "ActivationFunction.h"
#include "ClConvolveDllExport.h"

class ClConvolve_EXPORT FullyConnectedMaker : public LayerMaker2 {
public:
    int _numPlanes;
    int _imageSize;
    int _biased;
    ActivationFunction const*_activationFunction;
    FullyConnectedMaker() :
        _numPlanes(0),
        _imageSize(0),
        _activationFunction( new TanhActivation() ) {
    }
    FullyConnectedMaker *numPlanes(int numPlanes) {
        this->_numPlanes = numPlanes;
        return this;
    }    
    FullyConnectedMaker *imageSize(int imageSize) {
        this->_imageSize = imageSize;
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
    FullyConnectedMaker *fn(ActivationFunction const*_fn) {
        this->_activationFunction = _fn;
        return this;
    }
    static FullyConnectedMaker *instance() {
        return new FullyConnectedMaker();
    }
    virtual FullyConnectedMaker *clone() const {
        FullyConnectedMaker *thisClone = new FullyConnectedMaker();
        memcpy( thisClone, this, sizeof( FullyConnectedMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};



