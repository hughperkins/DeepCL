// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "LayerMaker.h"
#include "DeepCLDllExport.h"

class DeepCL_EXPORT NormalizationLayerMaker : public LayerMaker2 {
public:
    float _translate;
    float _scale;
    NormalizationLayerMaker() :
        _translate(0.0f),
        _scale( 1.0f ) {
    }
    NormalizationLayerMaker *translate( float _translate ) {
        this->_translate = _translate;
        return this;
    }
    NormalizationLayerMaker *scale( float _scale ) {
        this->_scale = _scale;
        return this;
    }
    static NormalizationLayerMaker *instance() {
        return new NormalizationLayerMaker();
    }
    virtual NormalizationLayerMaker *clone() const {
        NormalizationLayerMaker *thisClone = new NormalizationLayerMaker();
        memcpy( thisClone, this, sizeof( NormalizationLayerMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

