// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Propagate.h"

#define STATIC static
#define VIRTUAL virtual

class PropagateFc : public Propagate {
public:
    CLKernel *kernel1;
    CLKernel *kernel_reduce;
//    CLKernel *kernel_activate;
//    CLKernel *kPerElementAdd;
    CLKernel *kPerElementTiledAdd;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PropagateFc();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *outputWrapper );
    PropagateFc( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

