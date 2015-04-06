// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "LayerMaker.h"
#include "DeepCLDllExport.h"

class DeepCL_EXPORT ForceBackpropLayerMaker : public LayerMaker2 {
public:
    ForceBackpropLayerMaker() {
    }
    static ForceBackpropLayerMaker *instance() {
        return new ForceBackpropLayerMaker();
    }
    virtual ForceBackpropLayerMaker *clone() const {
        ForceBackpropLayerMaker *thisClone = new ForceBackpropLayerMaker();
        memcpy( thisClone, this, sizeof( ForceBackpropLayerMaker ) );
        return thisClone;
    }
    virtual Layer *createLayer( Layer *previousLayer );
};

