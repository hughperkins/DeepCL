#pragma once

#include "Propagate.h"

class Propagate4 : public Propagate {
public:
    CLKernel *kernel;
    int workgroupSize;
    int pixelsPerThread;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Propagate4();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *outputWrapper );
    Propagate4( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

