// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "ActivationFunction.h"
#include "DeepCLDllExport.h"

#include "LayerMaker.h"

class DeepCL_EXPORT ConvolutionalMaker : public LayerMaker2 {
public:
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    int _biased;
    ActivationFunction const *_activationFunction;
    ConvolutionalMaker() :
            _numFilters(0),
            _filterSize(0),
            _padZeros(false),
        _activationFunction( new TanhActivation() ) {
    }
    static ConvolutionalMaker *instance() {
        return new ConvolutionalMaker();
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
    ConvolutionalMaker *fn(ActivationFunction const*_fn) {
        this->_activationFunction = _fn;
        return this;
    }
    virtual ConvolutionalMaker *clone() const {
        ConvolutionalMaker *thisClone = new ConvolutionalMaker();
        memcpy( thisClone, this, sizeof( ConvolutionalMaker ) ); // this will copy the activationfunction pointer too
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

