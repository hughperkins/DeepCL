#pragma once

#include "Propagate.h"

class PropagateByInputPlane : public Propagate {
public:
    CLKernel *kernel;
    CLKernel *reduceSegments;
    CLKernel *repeatedAdd;
    CLKernel *activate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PropagateByInputPlane();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );
    PropagateByInputPlane( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );

    // [[[end]]]
};

