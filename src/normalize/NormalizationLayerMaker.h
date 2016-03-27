// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "layer/LayerMaker.h"
#include "DeepCLDllExport.h"

/// \brief Use to add a NormalizationLayer to a NeuralNet
///
/// A NormalizationLayer will normally be inserted as the second
/// layer in a network, after an InputLayer.  It can translate
/// and scale the input values.
PUBLICAPI
class DeepCL_EXPORT NormalizationLayerMaker : public LayerMaker2 {
public:
    float _translate;
    float _scale;
    PUBLICAPI NormalizationLayerMaker() :
        _translate(0.0f),
        _scale(1.0f) {
    }
//    NormalizationLayerMaker(float _translate, float _scale) :
//        _translate(_translate),
//        _scale(_scale) {
//    }
    PUBLICAPI NormalizationLayerMaker *translate(float _translate) {
        this->_translate = _translate;
        return this;
    }
    PUBLICAPI NormalizationLayerMaker *scale(float _scale) {
        this->_scale = _scale;
        return this;
    }
    PUBLICAPI static NormalizationLayerMaker *instance() {
        return new NormalizationLayerMaker();
    }
    virtual NormalizationLayerMaker *clone() const {
        return new NormalizationLayerMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

