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
#include "LayerDimensions.h"
#include "ClConvolveDllExport.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class ClConvolve_EXPORT Propagate {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    ActivationFunction const*fn;

    virtual void propagate( int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
        CLWrapper *resultsWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC Propagate *instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    STATIC Propagate *instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    STATIC int getNumImplementations();
    STATIC bool plausiblyOptimal( int index, int batchSize, LayerDimensions dim, ActivationFunction const*fn );
    STATIC Propagate *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    STATIC Propagate *instanceSpecific( std::string name, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const*fn );
    VIRTUAL float * propagate( int batchSize, float *inputData, float *filters, float *biases );
    VIRTUAL void propagate( int batchSize, float *inputData, float *filters, float *biases, float *results );

    // [[[end]]]

};



