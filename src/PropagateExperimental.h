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
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );
    PropagateExperimental( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );

    // [[[end]]]
};

