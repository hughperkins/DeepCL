// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "EasyCL.h"
#include "DropoutBackward.h"
#include "util/StatefulTimer.h"

#include "DropoutBackwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutBackwardCpu::DropoutBackwardCpu(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        DropoutBackward(cl, numPlanes, inputSize, dropRatio) {
}
VIRTUAL void DropoutBackwardCpu::backward(int batchSize, uchar *mask,  float *gradOutput, float *gradInput) {
    int totalLinearSize = batchSize * numPlanes * inputSize * inputSize;
    for(int i = 0; i < totalLinearSize; i++) {
        gradInput[i] = mask[i] == 1 ? gradOutput[i] : 0.0f;
    }
}
VIRTUAL void DropoutBackwardCpu::backward(int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, 
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("DropoutBackwardCpu::backward start");

    maskWrapper->copyToHost();
    gradOutputWrapper->copyToHost();

    uchar *mask = reinterpret_cast<uchar *>(maskWrapper->getHostArray());
    float *gradOutput = reinterpret_cast<float *>(gradOutputWrapper->getHostArray());
    float *gradInput = new float[ getInputNumElements(batchSize) ];

    backward(batchSize, mask, gradOutput, gradInput);

    float *gradInputHostArray = reinterpret_cast<float *>(gradInputWrapper->getHostArray());
    memcpy(gradInputHostArray, gradInput, sizeof(float) * getInputNumElements(batchSize) );
    gradInputWrapper->copyToDevice();

    delete[] gradInput;
    
    StatefulTimer::instance()->timeCheck("DropoutBackwardCpu::backward end");
}

