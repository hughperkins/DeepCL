// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "EasyCL.h"
#include "activate/ActivationFunction.h"
#include "conv/LayerDimensions.h"
#include "DeepCLDllExport.h"

using namespace std;

//inline float square(float value) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT Forward {
public:
    EasyCL *cl;
    LayerDimensions dim;

    virtual ~Forward() {}
    virtual void forward(int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
        CLWrapper *outputWrapper) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    Forward(EasyCL *cl, LayerDimensions layerDimensions);
    STATIC Forward *instance(EasyCL *cl, LayerDimensions dim);
    STATIC Forward *instanceTest(EasyCL *cl, LayerDimensions layerDimensions);
    STATIC int getNumImplementations();
    STATIC bool plausiblyOptimal(int index, int batchSize, LayerDimensions dim);
    STATIC Forward *instanceSpecific(int idx, EasyCL *cl, LayerDimensions layerDimensions);
    STATIC Forward *instanceSpecific(std::string name, EasyCL *cl, LayerDimensions layerDimensions);
    VIRTUAL int getOutputTotalSize(int batchSize);
    VIRTUAL void forward(int batchSize, float *inputData, float *filters, float *biases, float *output);

    // [[[end]]]

};



