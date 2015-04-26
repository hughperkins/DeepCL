// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "DropoutBackprop.h"
#include "StatefulTimer.h"

#include "DropoutBackpropCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutBackpropCpu::DropoutBackpropCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        DropoutBackprop( cl, numPlanes, inputImageSize, dropRatio ) {
}
VIRTUAL void DropoutBackpropCpu::backpropErrors( int batchSize, uchar *mask,  float *errors, float *errorsForUpstream ) {
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
        errorsForUpstream[i] = mask[i] == 1 ? errors[i] : 0.0f;
    }
}
VIRTUAL void DropoutBackpropCpu::backpropErrors( int batchSize, CLWrapper *maskWrapper, CLWrapper *errorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backpropErrors start" );

    maskWrapper->copyToHost();
    errorsWrapper->copyToHost();

    uchar *mask = reinterpret_cast<uchar *>( maskWrapper->getHostArray() );
    float *errors = reinterpret_cast<float *>( errorsWrapper->getHostArray() );
    float *errorsForUpstream = new float[ getInputSize( batchSize ) ];

    backpropErrors( batchSize, mask, errors, errorsForUpstream );

    float *errorsForUpstreamHostArray = reinterpret_cast<float *>( errorsForUpstreamWrapper->getHostArray() );
    memcpy( errorsForUpstreamHostArray, errorsForUpstream, sizeof(float) * getInputSize( batchSize ) );
    errorsForUpstreamWrapper->copyToDevice();

    delete[] errorsForUpstream;
    
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backpropErrors end" );
}

