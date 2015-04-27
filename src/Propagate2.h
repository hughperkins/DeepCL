#pragma once

#include "Propagate.h"

class Propagate2 : public Propagate {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Propagate2();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *outputWrapper );
    Propagate2( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]

};

