#pragma once

#include "BackpropErrors.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrors2 : public BackpropErrors {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropErrors2
    // cppfile: BackpropErrors2.cpp

    BackpropErrors2( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    VIRTUAL ~BackpropErrors2();
    VIRTUAL void backpropErrors( int batchSize,
    CLWrapper *resultsWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
    CLWrapper *errorsForUpstreamWrapper );

    // [[[end]]]
};

