// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "ActivationBackprop.h"
#include "StatefulTimer.h"
#include "ActivationFunction.h"

#include "ActivationBackpropCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationBackpropCpu::ActivationBackpropCpu( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const *fn ) :
        ActivationBackprop( cl, numPlanes, inputImageSize, fn ) {
}
VIRTUAL void ActivationBackpropCpu::backpropErrors( int batchSize,  float *errors, float *errorsForUpstream ) {
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
        errorsForUpstream[i] = fn->calcDerivative( errors[i] );
    }
}
VIRTUAL void ActivationBackpropCpu::backpropErrors( int batchSize, CLWrapper *errorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("ActivationBackpropCpu::backpropErrors start" );

    errorsWrapper->copyToHost();

    float *errors = reinterpret_cast<float *>( errorsWrapper->getHostArray() );
    float *errorsForUpstream = new float[ getInputSize( batchSize ) ];

    backpropErrors( batchSize, errors, errorsForUpstream );

    float *errorsForUpstreamHostArray = reinterpret_cast<float *>( errorsForUpstreamWrapper->getHostArray() );
    memcpy( errorsForUpstreamHostArray, errorsForUpstream, sizeof(float) * getInputSize( batchSize ) );
    errorsForUpstreamWrapper->copyToDevice();

    delete[] errorsForUpstream;
    
    StatefulTimer::instance()->timeCheck("ActivationBackpropCpu::backpropErrors end" );
}

