// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Propagate.h"

class PropagateFc_workgroupPerFilterPlane : public Propagate {
public:
    CLKernel *kernel1;
    CLKernel *kernel2;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PropagateFc_workgroupPerFilterPlane();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *outputWrapper );
    PropagateFc_workgroupPerFilterPlane( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );

    // [[[end]]]
};

