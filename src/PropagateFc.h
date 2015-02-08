#pragma once

#include "Propagate.h"

class PropagateFc : public Propagate {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PropagateFc();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );
    PropagateFc( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );

    // [[[end]]]
};

