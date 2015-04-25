// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"
#include "StatefulTimer.h"
#include "ActivationFunction.h"

#include "ActivationPropagateCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationPropagateCpu::ActivationPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        ActivationPropagate( cl, numPlanes, inputImageSize, fn ) {
}
VIRTUAL void ActivationPropagateCpu::propagate( int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
//    cout << "ActivationPropagateCpu::propagate( CLWrapper * )" << endl;

    inputWrapper->copyToHost();

    float *input = reinterpret_cast<float *>( inputWrapper->getHostArray() );
    float *output = new float[ getResultsSize( batchSize ) ];

    propagate( batchSize, input, output );

    float *outputHostArray = reinterpret_cast<float *>( outputWrapper->getHostArray() );
    memcpy( outputHostArray, output, sizeof(float) * getResultsSize( batchSize ) );

    outputWrapper->copyToDevice();

    delete[] output;
}
VIRTUAL void ActivationPropagateCpu::propagate( int batchSize, float *input, float *output ) {
//    float *output = new float[ getResultsSize( batchSize ) ];
//    cout << "ActivationPropagateCpu::propagate( float * )" << endl;
    StatefulTimer::instance()->timeCheck("ActivationPropagateCpu::propagate start" );
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
        output[i] = fn->calc( input[i] );
    }
    StatefulTimer::instance()->timeCheck("ActivationPropagateCpu::propagate end" );
//    return output;
}

