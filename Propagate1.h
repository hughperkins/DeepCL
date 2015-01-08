#pragma once

#include "Propagate.h"

class Propagate1 : public Propagate {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: Propagate1
    // cppfile: Propagate1.cpp

    Propagate1( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction *fn );
    VIRTUAL ~Propagate1();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );

    // [[[end]]]
};

