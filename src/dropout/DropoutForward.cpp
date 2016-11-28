// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/stringhelper.h"
#include "DropoutForwardCpu.h"
#include "DropoutForwardGpuNaive.h"

#include "DropoutForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutForward::DropoutForward(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        cl(cl),
        numPlanes(numPlanes),
        inputSize(inputSize),
        dropRatio(dropRatio),
        outputSize(inputSize) {
//    if(inputSize % dropoutSize != 0) {
//        throw runtime_error("inputSize should be an exact multiple of dropoutsize: " + toString(inputSize) + " " + toString(dropoutSize) );
//    }
}
STATIC DropoutForward *DropoutForward::instance(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
    return new DropoutForwardGpuNaive(cl, numPlanes, inputSize, dropRatio);
//    return new DropoutForwardCpu(cl, padZeros, numPlanes, inputSize, dropoutSize);
}
STATIC DropoutForward *DropoutForward::instanceForTest(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
    return new DropoutForwardCpu(cl, numPlanes, inputSize, dropRatio);
}
STATIC DropoutForward *DropoutForward::instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
    if(idx == 0) {
        return new DropoutForwardCpu(cl, numPlanes, inputSize, dropRatio);
    }
    if(idx == 1) {
        return new DropoutForwardGpuNaive(cl, numPlanes, inputSize, dropRatio);
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("DropoutForward::instanceSpecific idx not known: " + toString(idx) );
}
VIRTUAL void DropoutForward::forward(int batchSize, CLWrapper *masksWrapper, CLWrapper *inputData, CLWrapper *outputData) {
    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void DropoutForward::forward(int batchSize, unsigned char *masks, float *input, float *output) {
//    cout << "DropoutForward::forward(float *)" << endl;
    int inputLinearSize = getInputNumElements(batchSize);
    CLWrapper *masksWrapper = cl->wrap(inputLinearSize, masks);
    CLWrapper *inputWrapper = cl->wrap(inputLinearSize, input);
    CLWrapper *outputWrapper = cl->wrap(getOutputNumElements(batchSize), output);

    masksWrapper->copyToDevice();
    inputWrapper->copyToDevice();
    outputWrapper->createOnDevice();
    forward(batchSize, masksWrapper, inputWrapper, outputWrapper);
    outputWrapper->copyToHost();

    delete outputWrapper;
    delete inputWrapper;
    delete masksWrapper;
}
VIRTUAL int DropoutForward::getInputNumElements(int batchSize) {
    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int DropoutForward::getOutputNumElements(int batchSize) {
    return batchSize * numPlanes * outputSize * outputSize;
}


