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
#include "conv/Forward.h"
#include "conv/LayerDimensions.h"
#include "DeepCLDllExport.h"

using namespace std;

//inline float square(float value) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT ForwardAuto : public Forward {
public:
//    EasyCL *cl;
//    LayerDimensions dim;
//    ActivationFunction const*fn;

    int num;
    int *microseconds;
    bool *valid;
    int chosenIndex;
    Forward **instances;
    int nextIndex;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ForwardAuto(EasyCL *cl, LayerDimensions dim);
    VIRTUAL ~ForwardAuto();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper,
    CLWrapper *biasWrapper, CLWrapper *outputWrapper);

    // [[[end]]]

};



