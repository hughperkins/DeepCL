// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "activate/ActivationFunction.h"

#include "activate/ActivationForwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationForwardCpu::ActivationForwardCpu( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        ActivationForward( cl, numPlanes, inputImageSize, fn ) {
}
VIRTUAL void ActivationForwardCpu::forward( int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
//    cout << "ActivationForwardCpu::forward( CLWrapper * )" << endl;

    inputWrapper->copyToHost();

    float *input = reinterpret_cast<float *>( inputWrapper->getHostArray() );
    float *output = new float[ getOutputSize( batchSize ) ];

    forward( batchSize, input, output );

    float *outputHostArray = reinterpret_cast<float *>( outputWrapper->getHostArray() );
    memcpy( outputHostArray, output, sizeof(float) * getOutputSize( batchSize ) );

    outputWrapper->copyToDevice();

    delete[] output;
}
VIRTUAL void ActivationForwardCpu::forward( int batchSize, float *input, float *output ) {
//    float *output = new float[ getOutputSize( batchSize ) ];
//    cout << "ActivationForwardCpu::forward( float * )" << endl;
    StatefulTimer::instance()->timeCheck("ActivationForwardCpu::forward start" );
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
        output[i] = fn->calc( input[i] );
    }
    StatefulTimer::instance()->timeCheck("ActivationForwardCpu::forward end" );
//    return output;
}

