// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "EasyCL.h"

#include "util/StatefulTimer.h"

#include "DropoutForwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutForwardCpu::DropoutForwardCpu(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        DropoutForward(cl, numPlanes, inputSize, dropRatio) {
}
VIRTUAL void DropoutForwardCpu::forward(int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper) {
//    cout << "DropoutForwardCpu::forward(CLWrapper *)" << endl;

    inputWrapper->copyToHost();

    unsigned char *masks = reinterpret_cast<unsigned char *>(masksWrapper->getHostArray());
    float *input = reinterpret_cast<float *>(inputWrapper->getHostArray());
    float *output = new float[ getOutputNumElements(batchSize) ];

    forward(batchSize, masks, input, output);

    float *outputHostArray = reinterpret_cast<float *>(outputWrapper->getHostArray());
    memcpy(outputHostArray, output, sizeof(float) * getOutputNumElements(batchSize) );

    outputWrapper->copyToDevice();

    delete[] output;
}
VIRTUAL void DropoutForwardCpu::forward(int batchSize, unsigned char *masks, float *input, float *output) {
//    float *output = new float[ getOutputNumElements(batchSize) ];
//    cout << "DropoutForwardCpu::forward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("DropoutForwardCpu::forward start");
    int totalLinearSize = batchSize * numPlanes * inputSize * inputSize;
//    float inverseDropRatio = 1.0f / dropRatio; // since multiply faster than divide, just divide once
    for(int i = 0; i < totalLinearSize; i++) {
        output[i] = masks[i] == 1 ? input[i] : 0;
    }
    StatefulTimer::instance()->timeCheck("DropoutForwardCpu::forward end");
//    return output;
}

