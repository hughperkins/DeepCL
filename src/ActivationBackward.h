// Copyright Hugh Perkins 2014 hughperkins at gmail
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
class ActivationFunction;

class DeepCL_EXPORT ActivationBackward {
public:
    EasyCL *cl;

    const int numPlanes;
    const int inputImageSize;
    ActivationFunction const *fn;

    const int outputImageSize;

    virtual ~ActivationBackward() {}
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
    STATIC ActivationBackward *instance( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    STATIC ActivationBackward *instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn);
    STATIC ActivationBackward *instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    ActivationBackward( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);
    VIRTUAL void backward( int batchSize, float *inputs, float *gradOutput, float *gradInput );
    VIRTUAL void backward( int batchSize, CLWrapper *inputsWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper );

    // [[[end]]]
};

