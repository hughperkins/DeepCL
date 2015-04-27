#pragma once

#include "Propagate.h"

class PropagateExperimental : public Propagate {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PropagateExperimental();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *outputWrapper );
    PropagateExperimental( OpenCLHelper *cl, LayerDimensions dim );

    // [[[end]]]
};

