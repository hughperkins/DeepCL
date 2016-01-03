// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "layer/LayerMaker.h"
#include "DeepCLDllExport.h"

/// \brief use to create a RandomTranslations Layer
///
/// A RandomTranslations Layer translates the incoming data by a 
/// random amount when training is set to true, in the NeuralNet,
/// or by zero when training is set to false.
///
/// The size of the random translations is set by translateSize.
PUBLICAPI
class DeepCL_EXPORT RandomTranslationsMaker : public LayerMaker2 {
public:
    int _translateSize;
    PUBLICAPI RandomTranslationsMaker() :
        _translateSize(0) {
    }
    PUBLICAPI static RandomTranslationsMaker *instance() {
        return new RandomTranslationsMaker();
    }    
    PUBLICAPI RandomTranslationsMaker *translateSize(int _translateSize) {
        this->_translateSize = _translateSize;
        return this;
    }
    virtual RandomTranslationsMaker *clone() const {
        return new RandomTranslationsMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

