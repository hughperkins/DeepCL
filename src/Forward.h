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

class DeepCL_EXPORT Forward {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;

    virtual ~Forward() {}
    virtual void forward( int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
        CLWrapper *outputWrapper ) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Forward( OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC Forward *instance(OpenCLHelper *cl, LayerDimensions dim );
    STATIC Forward *instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC int getNumImplementations();
    STATIC bool plausiblyOptimal( int index, int batchSize, LayerDimensions dim );
    STATIC Forward *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions );
    STATIC Forward *instanceSpecific( std::string name, OpenCLHelper *cl, LayerDimensions layerDimensions );
    VIRTUAL float * forward( int batchSize, float *inputData, float *filters, float *biases );
    VIRTUAL void forward( int batchSize, float *inputData, float *filters, float *biases, float *output );

    // [[[end]]]

};



