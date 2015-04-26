// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

#include "DropoutBackpropCpu.h"
#include "DropoutBackpropGpuNaive.h"

#include "DropoutBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC DropoutBackprop *DropoutBackprop::instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutBackpropGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackprop *DropoutBackprop::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio) {
    return new DropoutBackpropCpu( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackprop *DropoutBackprop::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    if( idx == 0 ) {
        return new DropoutBackpropCpu( cl, numPlanes, inputImageSize, dropRatio );
    }
    if( idx == 1 ) {
        return new DropoutBackpropGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
    }
    throw runtime_error("DropoutBackprop::instanceSpecific, idx not known: " + toString( idx ) );
}
DropoutBackprop::DropoutBackprop( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        dropRatio( dropRatio ),
//        dropoutSizeSquared( dropoutSize * dropoutSize ),
        outputImageSize( inputImageSize ) {
//    if( inputImageSize % dropoutSize != 0 ) {
//        throw runtime_error("inputImageSize should be an exact multiple of dropoutsize: " + toString( inputImageSize ) + " " + toString(dropoutSize ) );
//    }
}
VIRTUAL int DropoutBackprop::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int DropoutBackprop::getResultsSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL void DropoutBackprop::backpropErrors( int batchSize, float *errors, float *errorsForUpstream ) {
//    cout << "DropoutBackprop::backpropErrors( float * )" << endl;
    StatefulTimer::instance()->timeCheck("DropoutBackprop::backpropErrors float->wrapper start" );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(batchSize), errors );
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( getInputSize(batchSize), errorsForUpstream );

    errorsWrapper->copyToDevice();

    backpropErrors( batchSize, errorsWrapper, errorsForUpstreamWrapper );

    errorsForUpstreamWrapper->copyToHost();

    delete errorsWrapper;
    delete errorsForUpstreamWrapper;
    StatefulTimer::instance()->timeCheck("DropoutBackprop::backpropErrors float->wrapper end" );
}
VIRTUAL void DropoutBackprop::backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *errorsForUpstreamWrapper ) {
    throw runtime_error("DropoutBackprop::backpropErrors wrappers not implemented" );
}

