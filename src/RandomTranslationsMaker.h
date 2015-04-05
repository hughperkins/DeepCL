// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "LayerMaker.h"
#include "DeepCLDllExport.h"

class DeepCL_EXPORT RandomTranslationsMaker : public LayerMaker2 {
public:
    int _translateSize;
    RandomTranslationsMaker() :
        _translateSize(0) {
    }
    static RandomTranslationsMaker *instance() {
        return new RandomTranslationsMaker();
    }    
    RandomTranslationsMaker *translateSize( int _translateSize ) {
        this->_translateSize = _translateSize;
        return this;
    }
    virtual RandomTranslationsMaker *clone() const {
        RandomTranslationsMaker *thisClone = new RandomTranslationsMaker();
        memcpy( thisClone, this, sizeof( RandomTranslationsMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

