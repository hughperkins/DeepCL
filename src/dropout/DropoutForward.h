// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class EasyCL;
class CLWrapper;

class DeepCL_EXPORT DropoutForward {
public:
    EasyCL *cl;

    const int numPlanes;
    const int inputImageSize;
    const float dropRatio;

    const int outputImageSize;

    virtual ~DropoutForward() {}
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
    DropoutForward( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    STATIC DropoutForward *instance( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    STATIC DropoutForward *instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    STATIC DropoutForward *instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    VIRTUAL void forward( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputData, CLWrapper *outputData );
    VIRTUAL void forward( int batchSize, unsigned char *masks, float *input, float *output );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);

    // [[[end]]]
};

