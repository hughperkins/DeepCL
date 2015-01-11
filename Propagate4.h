#pragma once

#include "Propagate.h"

class Propagate4 : public Propagate {
public:
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: Propagate4
    // cppfile: Propagate4.cpp

    Propagate4( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn );
    VIRTUAL ~Propagate4();
    VIRTUAL void propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper );

    // [[[end]]]
};

