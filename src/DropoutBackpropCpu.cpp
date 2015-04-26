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
VIRTUAL void DropoutBackpropCpu::backpropErrors( int batchSize,  float *errors, float *errorsForUpstream ) {
    memset( errorsForUpstream, 0, sizeof( float ) * getInputSize( batchSize ) );
//    for( int n = 0; n < batchSize; n++ ) {
//        for( int plane = 0; plane < numPlanes; plane++ ) {
//            for( int outputRow = 0; outputRow < outputImageSize; outputRow++ ) {
//                int inputRow = outputRow * dropoutSize;
//                for( int outputCol = 0; outputCol < outputImageSize; outputCol++ ) {
//                    int inputCol = outputCol * dropoutSize;
//                    int resultIndex = getResultIndex( n, plane, outputRow, outputCol );
//                    float error = errors[resultIndex];
//                    errorsForUpstream[ inputIndex ] = error;
//                }
//            }
//        }
//    }
}
VIRTUAL void DropoutBackpropCpu::backpropErrors( int batchSize, CLWrapper *errorsWrapper, 
        CLWrapper *errorsForUpstreamWrapper ) {
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backpropErrors start" );

    errorsWrapper->copyToHost();

    float *errors = reinterpret_cast<float *>( errorsWrapper->getHostArray() );
    float *errorsForUpstream = new float[ getInputSize( batchSize ) ];

    backpropErrors( batchSize, errors, errorsForUpstream );

    float *errorsForUpstreamHostArray = reinterpret_cast<float *>( errorsForUpstreamWrapper->getHostArray() );
    memcpy( errorsForUpstreamHostArray, errorsForUpstream, sizeof(float) * getInputSize( batchSize ) );
    errorsForUpstreamWrapper->copyToDevice();

    delete[] errorsForUpstream;
    
    StatefulTimer::instance()->timeCheck("DropoutBackpropCpu::backpropErrors end" );
}

