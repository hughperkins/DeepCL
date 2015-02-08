// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "OpenCLHelper.h"
#include "stringhelper.h"

#include "PoolingBackpropCpu.h"

#include "PoolingBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC PoolingBackprop *PoolingBackprop::instance( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) {
    return new PoolingBackpropCpu( cl, padZeros, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceForTest( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize) {
    return new PoolingBackpropCpu( cl, padZeros, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceSpecific( int idx, OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingBackpropCpu( cl, padZeros, numPlanes, inputBoardSize, poolingSize );
    }
    throw runtime_error("PoolingBackprop::instanceSpecific, idx not known: " + toString( idx ) );
}
PoolingBackprop::PoolingBackprop( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) :
        cl( cl ),
        padZeros( padZeros ),
        numPlanes( numPlanes ),
        inputBoardSize( inputBoardSize ),
        poolingSize( poolingSize ),
        outputBoardSize( padZeros ? ( inputBoardSize + poolingSize - 1 ) / poolingSize : inputBoardSize / poolingSize ) {
//    if( inputBoardSize % poolingSize != 0 ) {
//        throw runtime_error("inputBoardSize should be an exact multiple of poolingsize: " + toString( inputBoardSize ) + " " + toString(poolingSize ) );
//    }
}
VIRTUAL int PoolingBackprop::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputBoardSize * inputBoardSize;
}
VIRTUAL int PoolingBackprop::getResultsSize(int batchSize) {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL void PoolingBackprop::backpropErrors( int batchSize, float *errors, int *selectors, float *errorsForUpstream ) {
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(batchSize), errors );
    CLWrapper *selectorsWrapper = cl->wrap( getResultsSize(batchSize), selectors );
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( getInputSize(batchSize), errorsForUpstream );

    errorsWrapper->copyToDevice();

    backpropErrors( batchSize, errorsWrapper, selectorsWrapper, errorsForUpstreamWrapper );

    selectorsWrapper->copyToHost();
    errorsForUpstreamWrapper->copyToHost();

    delete errorsWrapper;
    delete selectorsWrapper;
    delete errorsForUpstreamWrapper;
}
VIRTUAL void PoolingBackprop::backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper, CLWrapper *errorsForUpstreamWrapper ) {
    throw runtime_error("PoolingBackprop::backpropErrors wrappers not implemented" );
}

