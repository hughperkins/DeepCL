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
VIRTUAL void ActivationBackpropCpu::backpropErrors( int batchSize, float *inputs, float *errors, float *errorsForUpstream ) {
    int totalLinearSize = batchSize * numPlanes * inputImageSize * inputImageSize;
    for( int i = 0; i < totalLinearSize; i++ ) {
//        cout << "input=" << inputs[i] << " deriv=" << fn->calcDerivative( inputs[i] )
//            << " error=" << errors[i];
        errorsForUpstream[i] = fn->calcDerivative( inputs[i] ) * errors[i];
//        cout << " errorsforupstream=" << errorsForUpstream[i] << endl;
    }
}
VIRTUAL void ActivationBackpropCpu::backpropErrors( int batchSize, CLWrapper *inputsWrapper,
         CLWrapper *errorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("ActivationBackpropCpu::backpropErrors start" );

    inputsWrapper->copyToHost();
    errorsWrapper->copyToHost();

    float *inputs = reinterpret_cast<float *>( inputsWrapper->getHostArray() );
    float *errors = reinterpret_cast<float *>( errorsWrapper->getHostArray() );
    float *errorsForUpstream = new float[ getInputSize( batchSize ) ];

    backpropErrors( batchSize, inputs, errors, errorsForUpstream );

    float *errorsForUpstreamHostArray = reinterpret_cast<float *>( errorsForUpstreamWrapper->getHostArray() );
    memcpy( errorsForUpstreamHostArray, errorsForUpstream, sizeof(float) * getInputSize( batchSize ) );
    errorsForUpstreamWrapper->copyToDevice();

    delete[] errorsForUpstream;
    
    StatefulTimer::instance()->timeCheck("ActivationBackpropCpu::backpropErrors end" );
}

