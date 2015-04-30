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

/// Use to create a convolutional layer
PUBLICAPI
class DeepCL_EXPORT ConvolutionalMaker : public LayerMaker2 {
public:
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    bool _biased;
//    ActivationFunction const *_activationFunction;
    PUBLICAPI ConvolutionalMaker() :
            _numFilters(0),
            _filterSize(0),
            _padZeros(false) {
    }
    PUBLICAPI static ConvolutionalMaker *instance() {
        return new ConvolutionalMaker();
    }    
    PUBLICAPI ConvolutionalMaker *numFilters(int numFilters) {
        this->_numFilters = numFilters;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *filterSize(int filterSize) {
        this->_filterSize = filterSize;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *padZeros( bool value ) {
        this->_padZeros = value;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *biased() {
        this->_biased = true;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *biased(bool _biased) {
        this->_biased = _biased;
        return this;
    }    
//    PUBLICAPI ConvolutionalMaker *tanh() {
//        delete this->_activationFunction;
//        this->_activationFunction = new TanhActivation();
//        return this;
//    }
//    PUBLICAPI ConvolutionalMaker *relu() {
//        delete this->_activationFunction;
//        this->_activationFunction = new ReluActivation();
//        return this;
//    }
//    PUBLICAPI ConvolutionalMaker *sigmoid() {
//        delete this->_activationFunction;
//        this->_activationFunction = new SigmoidActivation();
//        return this;
//    }
//    PUBLICAPI ConvolutionalMaker *linear() {
//        delete this->_activationFunction;
//        this->_activationFunction = new LinearActivation();
//        return this;
//    }
//    PUBLICAPI ConvolutionalMaker *fn(ActivationFunction const*_fn) {
//        this->_activationFunction = _fn;
//        return this;
//    }
    virtual ConvolutionalMaker *clone() const {
        ConvolutionalMaker *thisClone = new ConvolutionalMaker();
        memcpy( thisClone, this, sizeof( ConvolutionalMaker ) ); // this will copy the activationfunction pointer too
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

