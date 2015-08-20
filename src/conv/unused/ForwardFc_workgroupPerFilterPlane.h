// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Forward.h"

class ForwardFc_workgroupPerFilterPlane : public Forward {
public:
    CLKernel *kernel1;
    CLKernel *kernel2;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardFc_workgroupPerFilterPlane();
    VIRTUAL void forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper);
    ForwardFc_workgroupPerFilterPlane(EasyCL *cl, LayerDimensions dim);

    // [[[end]]]
};

