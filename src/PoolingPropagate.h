// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class OpenCLHelper;
class CLWrapper;

class DeepCL_EXPORT PoolingPropagate {
public:
    OpenCLHelper *cl;

    const bool padZeros;
    const int numPlanes;
    const int inputImageSize;
    const int poolingSize;

    const int outputImageSize;

    virtual ~PoolingPropagate() {}
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
    PoolingPropagate( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize );
    STATIC PoolingPropagate *instance( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize );
    STATIC PoolingPropagate *instanceForTest( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize );
    STATIC PoolingPropagate *instanceSpecific( int idx, OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize );
    VIRTUAL void propagate( int batchSize, CLWrapper *inputData, CLWrapper *selectors, CLWrapper *outputData );
    VIRTUAL void propagate( int batchSize, float *input, int *selectors, float *output );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);

    // [[[end]]]
};

