// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstring>

#include "layer/LayerMaker.h"
#include "DeepCLDllExport.h"

/// \brief Use to create a RandomPatches Layer
///
/// A RandomPatches layer cuts a patch from the previous layer
/// and passes it to the next layer.
///
/// The size of the patch is configurable, and the location
/// is random.  When the NeuralNet, containing this layer,
/// is set to training=false, then the patch is cut from
/// the centre
PUBLICAPI
class DeepCL_EXPORT RandomPatchesMaker : public LayerMaker2 {
public:
    int _patchSize;
    PUBLICAPI RandomPatchesMaker() :
        _patchSize(0) {
    }
    PUBLICAPI RandomPatchesMaker *patchSize(int _patchSize) {
        this->_patchSize = _patchSize;
        return this;
    }
    PUBLICAPI static RandomPatchesMaker *instance() {
        return new RandomPatchesMaker();
    }
    virtual RandomPatchesMaker *clone() const {
        return new RandomPatchesMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

