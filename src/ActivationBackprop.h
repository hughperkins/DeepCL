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
class ActivationFunction;

class DeepCL_EXPORT ActivationBackprop {
public:
    OpenCLHelper *cl;

    const int numPlanes;
    const int inputImageSize;
    ActivationFunction const *fn;

    const int outputImageSize;

    virtual ~ActivationBackprop() {}
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
    STATIC ActivationBackprop *instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    STATIC ActivationBackprop *instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn);
    STATIC ActivationBackprop *instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    ActivationBackprop( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn );
    VIRTUAL int getInputSize( int batchSize );
    VIRTUAL int getOutputSize(int batchSize);
    VIRTUAL void backpropErrors( int batchSize, float *inputs, float *errors, float *gradInput );
    VIRTUAL void backpropErrors( int batchSize, CLWrapper *inputsWrapper, CLWrapper *errorsWrapper, CLWrapper *gradInputWrapper );

    // [[[end]]]
};

