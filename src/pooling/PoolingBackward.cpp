// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "EasyCL.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

#include "PoolingBackwardCpu.h"
#include "PoolingBackwardGpuNaive.h"

#include "PoolingBackward.h"

using namespace std;
using namespace easycl;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC PoolingBackward *PoolingBackward::instance(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
    return new PoolingBackwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize);
}
STATIC PoolingBackward *PoolingBackward::instanceForTest(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
    return new PoolingBackwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
}
STATIC PoolingBackward *PoolingBackward::instanceSpecific(int idx, EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
    if(idx == 0) {
        return new PoolingBackwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
    }
    if(idx == 1) {
        return new PoolingBackwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize);
    }
    throw runtime_error("PoolingBackward::instanceSpecific, idx not known: " + toString(idx) );
}
PoolingBackward::PoolingBackward(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) :
        cl(cl),
        padZeros(padZeros),
        numPlanes(numPlanes),
        inputSize(inputSize),
        poolingSize(poolingSize),
//        poolingSizeSquared(poolingSize * poolingSize),
        outputSize(padZeros ? (inputSize + poolingSize - 1) / poolingSize : inputSize / poolingSize) {
//    if(inputSize % poolingSize != 0) {
//        throw runtime_error("inputSize should be an exact multiple of poolingsize: " + toString(inputSize) + " " + toString(poolingSize) );
//    }
}
VIRTUAL int PoolingBackward::getInputNumElements(int batchSize) {
    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int PoolingBackward::getOutputNumElements(int batchSize) {
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL void PoolingBackward::backward(int batchSize, float *gradOutput, int *selectors, float *gradInput) {
//    cout << "PoolingBackward::backward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("PoolingBackward::backward float->wrapper start");
    CLWrapper *gradOutputWrapper = cl->wrap(getOutputNumElements(batchSize), gradOutput);
    CLWrapper *selectorsWrapper = cl->wrap(getOutputNumElements(batchSize), selectors);
    CLWrapper *gradInputWrapper = cl->wrap(getInputNumElements(batchSize), gradInput);

    gradOutputWrapper->copyToDevice();
    selectorsWrapper->copyToDevice();

    backward(batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper);

    selectorsWrapper->copyToHost();
    gradInputWrapper->copyToHost();

    delete gradOutputWrapper;
    delete selectorsWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("PoolingBackward::backward float->wrapper end");
}
VIRTUAL void PoolingBackward::backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper, CLWrapper *gradInputWrapper) {
    throw runtime_error("PoolingBackward::backward wrappers not implemented");
}

