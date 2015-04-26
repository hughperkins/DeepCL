// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class ActivationFunction;
class OpenCLHelper;
class CLWrapper;

class DeepCL_EXPORT ActivationPropagate {
public:
    OpenCLHelper *cl;

    const int numPlanes;
    const int inputImageSize;

    const int outputImageSize;

    ActivationFunction const*fn;

    virtual ~ActivationPropagate() {}
    inline int getInputIndex( int n, int plane, int row, int col ) {
        return ( ( n
            * numPlanes + plane )
            * inputImageSize + row )
            * inputImageSize + col;
    }
    inline int getResultIndex( int n, int plane, int row, int col ) {
        return ( ( n
            * numPlanes + plane )
            * outputImageSize + row )
            * outputImageSize + col;
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationPropagate( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationPropagate *instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationPropagate *instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationPropagate *instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    VIRTUAL void propagate( int batchSize, CLWrapper *inputData, CLWrapper *outputData );
    VIRTUAL void propagate( int batchSize, float *input, float *output );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);

    // [[[end]]]
};

