#pragma once

#include "Forward.h"

class ForwardExperimental : public Forward {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardExperimental();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper );
    ForwardExperimental( EasyCL *cl, LayerDimensions dim );

    // [[[end]]]
};

