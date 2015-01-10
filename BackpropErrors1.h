#pragma once

#include "BackpropErrors.h"

#define STATIC static
#define VIRTUAL virtual

class BackpropErrors1 : public BackpropErrors {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: BackpropErrors1
    // cppfile: BackpropErrors1.cpp

    BackpropErrors1( OpenCLHelper *cl, LayerDimensions dim );
    VIRTUAL ~BackpropErrors1();
    VIRTUAL void backpropErrors( int batchSize,
    CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper, CLWrapper *errorsWrapper,
    CLWrapper *errorsForUpstreamWrapper );

    // [[[end]]]
};

