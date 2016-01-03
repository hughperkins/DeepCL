// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>
#include <iostream>

#include "DeepCLDllExport.h"
#include "layer/LayerMaker.h"

class Layer;

/// \brief Use to create an InputLayer, which can be added to a NeuralNet
///
/// This is the first layer in any network, and can receive one batch of data,
/// that we want to forward-forward
PUBLICAPI
class DeepCL_EXPORT InputLayerMaker : public LayerMaker2 {
public:
    int _numPlanes;
    int _imageSize;
    PUBLICAPI InputLayerMaker() :
//            LayerMaker(net, 0),
            _numPlanes(0),
            _imageSize(0) {
    }
    PUBLICAPI InputLayerMaker *numPlanes(int _numPlanes) {
        this->_numPlanes = _numPlanes;
        return this;
    }    
    PUBLICAPI InputLayerMaker *imageSize(int _imageSize) {
        this->_imageSize = _imageSize;
        return this;
    }    
    PUBLICAPI static InputLayerMaker *instance() {
        return new InputLayerMaker();
    }    
    virtual InputLayerMaker *clone() const {
        return new InputLayerMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

