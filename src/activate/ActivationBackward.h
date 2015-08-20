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
    const int inputSize;
    ActivationFunction const *fn;

    const int outputSize;

    virtual ~ActivationBackward() {}
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
    STATIC ActivationBackward *instance(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn);
    STATIC ActivationBackward *instanceForTest(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn);
    STATIC ActivationBackward *instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn);
    ActivationBackward(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn);
    VIRTUAL int getInputNumElements(int batchSize);
    VIRTUAL int getOutputNumElements(int batchSize);
    VIRTUAL void backward(int batchSize, float *inputs, float *gradOutput, float *gradInput);
    VIRTUAL void backward(int batchSize, CLWrapper *inputsWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper);

    // [[[end]]]
};

