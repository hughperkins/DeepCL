// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "LayerMaker.h"
#include "DeepCLDllExport.h"

PUBLICAPI
class DeepCL_EXPORT RandomPatchesMaker : public LayerMaker2 {
public:
    int _patchSize;
    PUBLICAPI RandomPatchesMaker() :
        _patchSize(0) {
    }
    PUBLICAPI RandomPatchesMaker *patchSize( int _patchSize ) {
        this->_patchSize = _patchSize;
        return this;
    }
    PUBLICAPI static RandomPatchesMaker *instance() {
        return new RandomPatchesMaker();
    }
    virtual RandomPatchesMaker *clone() const {
        RandomPatchesMaker *thisClone = new RandomPatchesMaker();
        memcpy( thisClone, this, sizeof( RandomPatchesMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

