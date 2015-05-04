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
class EasyCL;
class CLWrapper;

class DeepCL_EXPORT ActivationForward {
public:
    EasyCL *cl;

    const int numPlanes;
    const int inputImageSize;

    const int outputImageSize;

    ActivationFunction const*fn;

    virtual ~ActivationForward() {}
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
    ActivationForward( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationForward *instance( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationForward *instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    STATIC ActivationForward *instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn );
    VIRTUAL void forward( int batchSize, CLWrapper *inputData, CLWrapper *outputData );
    VIRTUAL void forward( int batchSize, float *input, float *output );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);

    // [[[end]]]
};

