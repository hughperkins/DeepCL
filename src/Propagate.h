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
#include "DeepCLDllExport.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT Propagate {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;

    virtual ~Propagate() {}
    virtual void propagate( int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
        CLWrapper *outputWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC Propagate *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC Propagate *instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC int getNumImplementations();
    STATIC bool plausiblyOptimal( int index, int batchSize, LayerDimensions dim );
    STATIC Propagate *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC Propagate *instanceSpecific( std::string name, OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL float * propagate( int batchSize, float *inputData, float *filters, float *biases );
    VIRTUAL void propagate( int batchSize, float *inputData, float *filters, float *biases, float *output );

    // [[[end]]]

};



