// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/LayerMaker.h"
#include "DeepCLDllExport.h"

/// \brief Use to create a dropout layer
class DeepCL_EXPORT DropoutMaker : public LayerMaker2 {
public:
    float _dropRatio; // 0.0 -> 1.0
    DropoutMaker() :
        _dropRatio(0.5f) {
    }
    DropoutMaker *dropRatio(float _dropRatio) {
        this->_dropRatio = _dropRatio;
        return this;
    }
    static DropoutMaker *instance() {
        return new DropoutMaker();
    }
    virtual DropoutMaker *clone() const {
        return new DropoutMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};


