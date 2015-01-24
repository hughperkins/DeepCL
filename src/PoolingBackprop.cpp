// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "OpenCLHelper.h"
#include "stringhelper.h"

#include "PoolingBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC PoolingBackprop *PoolingBackprop::instance( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) {
    return new PoolingBackpropCpu( cl, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize) {
    return new PoolingBackpropCpu( cl, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingBackpropCpu( cl, numPlanes, inputBoardSize, poolingSize );
    }
    throw runtime_error("PoolingBackprop::instanceSpecific, idx not known: " + toString( idx ) );
}
PoolingBackprop::PoolingBackprop( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputBoardSize( inputBoardSize ),
        poolingSize( poolingSize ) {
}
VIRTUAL void PoolingBackprop::backpropErrors( int batchSize, float *errors, int *selectors, float *errorsForUpstream ) {
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(batchSize), errors );
    CLWrapper *selectorsWrapper = cl->wrap( getResultsSize(batchSize), selectors );
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( getInputSize(batchSize), errorsForUpstream );

    errorsWrapper->copyToDevice();

    backpropErrors( batchSize, errorsWrapper, selectorsWrapper, errorsForUpstreamWrapper );

    selectorsWrapper->copyToHost();
    errorsForUpstreamWrapper->copyToHost();

    delete selectorsWrapper;
    delete errorsForUpstreamWrapper;
}
VIRTUAL void PoolingBackprop::backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper, CLWrapper *errorsForUpstreamWrapper ) {
    throw runtime_error("PoolingBackprop::backpropErrors wrappers not implemented" );
}

