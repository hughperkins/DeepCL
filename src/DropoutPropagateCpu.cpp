// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"

#include "StatefulTimer.h"

#include "DropoutPropagateCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutPropagateCpu::DropoutPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        DropoutPropagate( cl, numPlanes, inputImageSize, dropRatio ) {
}
VIRTUAL void DropoutPropagateCpu::propagate( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
//    cout << "DropoutPropagateCpu::propagate( CLWrapper * )" << endl;

    inputWrapper->copyToHost();

    unsigned char *masks = reinterpret_cast<unsigned char *>( masksWrapper->getHostArray() );
    float *input = reinterpret_cast<float *>( inputWrapper->getHostArray() );
    float *output = new float[ getResultsSize( batchSize ) ];

    propagate( batchSize, masks, input, output );

    float *outputHostArray = reinterpret_cast<float *>( outputWrapper->getHostArray() );
    memcpy( outputHostArray, output, sizeof(float) * getResultsSize( batchSize ) );

    outputWrapper->copyToDevice();

    delete[] output;
}
VIRTUAL void DropoutPropagateCpu::propagate( int batchSize, unsigned char *masks, float *input, float *output ) {
//    float *output = new float[ getResultsSize( batchSize ) ];
//    cout << "DropoutPropagateCpu::propagate( float * )" << endl;
    StatefulTimer::instance()->timeCheck("DropoutPropagateCpu::propagate start" );
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
//    float inverseDropRatio = 1.0f / dropRatio; // since multiply faster than divide, just divide once
    for( int i = 0; i < totalLinearSize; i++ ) {
        output[i] = masks[i] == 1 ? input[i] : 0;
    }
    StatefulTimer::instance()->timeCheck("DropoutPropagateCpu::propagate end" );
//    return output;
}

