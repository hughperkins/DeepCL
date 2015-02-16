#pragma once

#include "Propagate.h"

class Propagate3_unfactorized : public Propagate {
public:
    CLKernel *kernel;
    CLKernel *repeatedAdd;
    CLKernel *activate;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~Propagate3_unfactorized();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );
    Propagate3_unfactorized( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );

    // [[[end]]]
};

