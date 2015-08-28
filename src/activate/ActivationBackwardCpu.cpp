// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "EasyCL.h"
#include "activate/ActivationBackward.h"
#include "util/StatefulTimer.h"
#include "activate/ActivationFunction.h"

#include "activate/ActivationBackwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationBackwardCpu::ActivationBackwardCpu(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn) :
        ActivationBackward(cl, numPlanes, inputSize, fn) {
}
VIRTUAL void ActivationBackwardCpu::backward(int batchSize, float *outputs, float *gradOutput, float *gradInput) {
    int totalLinearSize = batchSize * numPlanes * inputSize * inputSize;
    for(int i = 0; i < totalLinearSize; i++) {
//        cout << "input=" << inputs[i] << " deriv=" << fn->calcDerivative(inputs[i])
//            << " error=" << errors[i];
        gradInput[i] = fn->calcDerivative(outputs[i]) * gradOutput[i];
        cout << " gradInput=" << gradInput[i] << endl;
    }
}
VIRTUAL void ActivationBackwardCpu::backward(int batchSize, 
        CLWrapper *outputWrapper,
         CLWrapper *gradOutputWrapper, 
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("ActivationBackwardCpu::backward start");

    outputWrapper->copyToHost();
    gradOutputWrapper->copyToHost();

    float *outputs = reinterpret_cast<float *>(outputWrapper->getHostArray());
    float *gradOutput = reinterpret_cast<float *>(gradOutputWrapper->getHostArray());
    float *gradInput = new float[ getInputNumElements(batchSize) ];
    for(int i = 0; i < 4; i++) {
        cout << "i=" << i << " outputs=" << outputs[i] << " gradOutput=" << gradOutput[i] << endl;
    }

    backward(batchSize, outputs, gradOutput, gradInput);

    float *gradInputHostArray = reinterpret_cast<float *>(gradInputWrapper->getHostArray());
    memcpy(gradInputHostArray, gradInput, sizeof(float) * getInputNumElements(batchSize) );
    gradInputWrapper->copyToDevice();

    for(int i = 0; i < 4; i++) {
        cout << "i=" << i << " gradInput=" << gradInput[i] << endl;
    }

    delete[] gradInput;
    
    StatefulTimer::instance()->timeCheck("ActivationBackwardCpu::backward end");
}

