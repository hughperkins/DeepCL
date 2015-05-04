#pragma once

#include "Forward.h"

class Forward2 : public Forward {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Forward2();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper );
    Forward2( EasyCL *cl, LayerDimensions dim );

    // [[[end]]]

};

