// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Propagate.h"

#define STATIC static
#define VIRTUAL virtual

class PropagateCpu : public Propagate {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PropagateCpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );
    VIRTUAL void propagate( int batchSize, CLWrapper *inputDataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *outputWrapper );
    VIRTUAL float *propagate( int batchSize, float *inputData, float *weights, float *biasWeights );

    // [[[end]]]
};

