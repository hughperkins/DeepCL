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

class DeepCL_EXPORT DropoutBackward {
public:
    EasyCL *cl;

    const int numPlanes;
    const int inputImageSize;
    const float dropRatio;

    const int outputImageSize;

    virtual ~DropoutBackward() {}
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
    STATIC DropoutBackward *instance( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    STATIC DropoutBackward *instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio);
    STATIC DropoutBackward *instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    DropoutBackward( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);
    VIRTUAL void backward( int batchSize, uchar *mask, float *gradOutput, float *gradInput );
    VIRTUAL void backward( int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper );

    // [[[end]]]
};

