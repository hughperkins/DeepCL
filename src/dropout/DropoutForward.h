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
    const int inputSize;
    const float dropRatio;

    const int outputSize;

    virtual ~DropoutForward() {}
    inline int getInputIndex(int n, int plane, int row, int col) {
        return (( n
            * numPlanes + plane)
            * inputSize + row)
            * inputSize + col;
    }
    inline int getResultIndex(int n, int plane, int row, int col) {
        return (( n
            * numPlanes + plane)
            * outputSize + row)
            * outputSize + col;
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    DropoutForward(EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    STATIC DropoutForward *instance(EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    STATIC DropoutForward *instanceForTest(EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    STATIC DropoutForward *instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, float dropRatio);
    VIRTUAL void forward(int batchSize, CLWrapper *masksWrapper, CLWrapper *inputData, CLWrapper *outputData);
    VIRTUAL void forward(int batchSize, unsigned char *masks, float *input, float *output);
    VIRTUAL int getInputNumElements(int batchSize);
    VIRTUAL int getOutputNumElements(int batchSize);

    // [[[end]]]
};

