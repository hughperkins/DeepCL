// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>

#include "EasyCL.h"
#include "activate/ActivationFunction.h"
#include "conv/LayerDimensions.h"

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT Backward {
public:
    EasyCL *cl;
    LayerDimensions dim;
//    ActivationFunction const *upstreamFn;

    virtual ~Backward() {}
    virtual void backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput) = 0;

    // [[[cog
    // import cog_addheaders    
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC Backward *instance(EasyCL *cl, LayerDimensions dim);
    STATIC Backward *instanceForTest(EasyCL *cl, LayerDimensions layerDimensions);
    STATIC Backward *instanceSpecific(int idx, EasyCL *cl, LayerDimensions layerDimensions);
    Backward(EasyCL *cl, LayerDimensions layerDimensions);
    STATIC int getNumImplementations();
    STATIC bool plausiblyOptimal(int index, int batchSize, LayerDimensions dim);
    VIRTUAL float * backward(int batchSize, float *input, float *gradOutput, float *filters);

    // [[[end]]]
};

