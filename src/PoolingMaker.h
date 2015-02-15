// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "LayerMaker.h"
#include "ClConvolveDllExport.h"

class ClConvolve_EXPORT PoolingMaker : public LayerMaker2 {
public:
//    Layer *previousLayer;
    int _poolingSize;
    bool _padZeros;
    PoolingMaker() :
        _poolingSize( 2 ),
        _padZeros( false ) {
    }
    PoolingMaker *poolingSize( int _poolingSize ) {
        this->_poolingSize = _poolingSize;
        return this;
    }
    PoolingMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }
    static PoolingMaker *instance() {
        return new PoolingMaker();
    }
    virtual PoolingMaker *clone() const {
        PoolingMaker *thisClone = new PoolingMaker();
        memcpy( thisClone, this, sizeof( PoolingMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};


