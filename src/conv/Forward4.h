#pragma once

#include "Forward.h"

class Forward4 : public Forward {
public:
    CLKernel *kernel;
    int workgroupSize;
    int pixelsPerThread;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward4();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper );
    Forward4( EasyCL *cl, LayerDimensions dim );

    // [[[end]]]
};

