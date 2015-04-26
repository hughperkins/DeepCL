// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "Propagate.h"
#include "LayerDimensions.h"
#include "DeepCLDllExport.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT PropagateAuto : public Propagate {
public:
//    OpenCLHelper *cl;
//    LayerDimensions dim;
//    ActivationFunction const*fn;

    int num;
    int *milliseconds;
    bool *valid;
    int chosenIndex;
    Propagate **instances;
    int nextIndex;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PropagateAuto( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );
    VIRTUAL ~PropagateAuto();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper,
    CLWrapper *biasWeightsWrapper, CLWrapper *outputWrapper );

    // [[[end]]]

};



