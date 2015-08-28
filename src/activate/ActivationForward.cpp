// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/stringhelper.h"
#include "activate/ActivationForwardCpu.h"
#include "activate/ActivationForwardGpuNaive.h"

#include "activate/ActivationForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationForward::ActivationForward(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) :
        cl(cl),
        numPlanes(numPlanes),
        inputSize(inputSize),
        outputSize(inputSize),
        fn(fn) {
}
STATIC ActivationForward *ActivationForward::instance(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) {
    return new ActivationForwardGpuNaive(cl, numPlanes, inputSize, fn);
//    return new ActivationForwardCpu(cl, numPlanes, inputSize);
}
STATIC ActivationForward *ActivationForward::instanceForTest(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) {
    return new ActivationForwardCpu(cl, numPlanes, inputSize, fn);
}
STATIC ActivationForward *ActivationForward::instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) {
    if(idx == 0) {
        return new ActivationForwardCpu(cl, numPlanes, inputSize, fn);
    }
    if(idx == 1) {
        return new ActivationForwardGpuNaive(cl, numPlanes, inputSize, fn);
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("ActivationForward::instanceSpecific idx not known: " + toString(idx) );
}
VIRTUAL void ActivationForward::forward(int batchSize, CLWrapper *inputData, CLWrapper *outputData) {
    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void ActivationForward::forward(int batchSize, float *input, float *output) {
//    cout << "ActivationForward::forward(float *)" << endl;
    CLWrapper *inputWrapper = cl->wrap(getInputNumElements(batchSize), input);
    CLWrapper *outputWrapper = cl->wrap(getOutputNumElements(batchSize), output);

    inputWrapper->copyToDevice();
    outputWrapper->createOnDevice();
    forward(batchSize, inputWrapper, outputWrapper);
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
}
VIRTUAL int ActivationForward::getInputNumElements(int batchSize) {
    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int ActivationForward::getOutputNumElements(int batchSize) {
    return batchSize * numPlanes * outputSize * outputSize;
}

