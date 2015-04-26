// Copyright Hugh Perkins 2015 hughperkins at gmail
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

class DeepCL_EXPORT DropoutPropagate {
public:
    OpenCLHelper *cl;

    const int numPlanes;
    const int inputImageSize;

    const int outputImageSize;

    virtual ~DropoutPropagate() {}
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
    DropoutPropagate( OpenCLHelper *cl, int numPlanes, int inputImageSize );
    STATIC DropoutPropagate *instance( OpenCLHelper *cl, int numPlanes, int inputImageSize );
    STATIC DropoutPropagate *instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize );
    STATIC DropoutPropagate *instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize );
    VIRTUAL void propagate( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputData, CLWrapper *outputData );
    VIRTUAL void propagate( int batchSize, unsigned char *masks, float *input, float *output );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getResultsSize(int batchSize);

    // [[[end]]]
};

