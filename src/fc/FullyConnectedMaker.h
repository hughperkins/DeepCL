// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "layer/LayerMaker.h"
#include "activate/ActivationFunction.h"
#include "weights/OriginalInitializer.h"

#include "DeepCLDllExport.h"

/// \brief Use to create a fully-connected layer
PUBLICAPI
class DeepCL_EXPORT FullyConnectedMaker : public LayerMaker2 {
public:
    int _numPlanes;
    int _imageSize;
    bool _biased;
    WeightsInitializer *_weightsInitializer;

    PUBLICAPI FullyConnectedMaker() :
        _numPlanes(0),
        _imageSize(0),
        _biased(true),
        _weightsInitializer(new OriginalInitializer()) {
    }
    FullyConnectedMaker *weightsInitializer(WeightsInitializer *weightsInitializer) {
        this->_weightsInitializer = weightsInitializer;
        return this;
    }    
    PUBLICAPI FullyConnectedMaker *numPlanes(int numPlanes) {
        this->_numPlanes = numPlanes;
        return this;
    }    
    PUBLICAPI FullyConnectedMaker *imageSize(int imageSize) {
        this->_imageSize = imageSize;
        return this;
    }
    PUBLICAPI FullyConnectedMaker *biased() {
        this->_biased = true;
        return this;
    }    
    FullyConnectedMaker *biased(bool _biased) {
        this->_biased = _biased;
        return this;
    }    
    PUBLICAPI static FullyConnectedMaker *instance() {
        return new FullyConnectedMaker();
    }
    virtual FullyConnectedMaker *clone() const {
        return new FullyConnectedMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};



