// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "activate/ActivationFunction.h"
#include "layer/LayerMaker.h"
#include "weights/OriginalInitializer.h"

#include "DeepCLDllExport.h"

/// Use to create a convolutional layer
PUBLICAPI
class DeepCL_EXPORT ConvolutionalMaker : public LayerMaker2 {
public:
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    bool _biased;
    WeightsInitializer *_weightsInitializer;

    PUBLICAPI ConvolutionalMaker() :
            _numFilters(0),
            _filterSize(0),
            _padZeros(false),
            _biased(true),
            _weightsInitializer(new OriginalInitializer()) { // will leak slightly, but hopefully not much
    }
    PUBLICAPI static ConvolutionalMaker *instance() {
        return new ConvolutionalMaker();
    }
    ConvolutionalMaker *weightsInitializer(WeightsInitializer *weightsInitializer) {
        this->_weightsInitializer = weightsInitializer;
        return this;
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
    PUBLICAPI ConvolutionalMaker *padZeros(bool value) {
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
    virtual ConvolutionalMaker *clone() const {
        return new ConvolutionalMaker(*this); // this will copy the activationfunction pointer too
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

