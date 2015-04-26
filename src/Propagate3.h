#pragma once

#include "Propagate.h"

class Propagate3 : public Propagate {
public:
    CLKernel *kernel;
    CLKernel *repeatedAdd;
    CLKernel *activate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Propagate3();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *outputWrapper );
    Propagate3( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

