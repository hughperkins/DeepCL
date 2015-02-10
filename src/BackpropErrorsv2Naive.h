#pragma once

#include "BackpropErrorsv2.h"
#include "OpenCLHelper.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrorsv2Naive : public BackpropErrorsv2 {
public:
    CLKernel *kernel;
    CLKernel *broadcastMultiply;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropErrorsv2Naive();
    VIRTUAL void backpropErrors( int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *errorsWrapper, CLWrapper *weightsWrapper,
    CLWrapper *errorsForUpstreamWrapper );
    BackpropErrorsv2Naive( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *upstreamFn );

    // [[[end]]]
};

