// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "PoolingPropagateCpu.h"
#include "PoolingPropagateGpuNaive.h"

#include "PoolingPropagate.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingPropagate::PoolingPropagate( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) :
        cl( cl ),
        padZeros( padZeros ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        poolingSize( poolingSize ),
        outputImageSize( padZeros ? ( inputImageSize + poolingSize - 1 ) / poolingSize : inputImageSize / poolingSize ) {
//    if( inputImageSize % poolingSize != 0 ) {
//        throw runtime_error("inputImageSize should be an exact multiple of poolingsize: " + toString( inputImageSize ) + " " + toString(poolingSize ) );
//    }
}
STATIC PoolingPropagate *PoolingPropagate::instance( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    return new PoolingPropagateGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
//    return new PoolingPropagateCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingPropagate *PoolingPropagate::instanceForTest( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    return new PoolingPropagateGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingPropagate *PoolingPropagate::instanceSpecific( int idx, OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingPropagateCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    if( idx == 1 ) {
        return new PoolingPropagateGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("PoolingPropagate::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void PoolingPropagate::propagate( int batchSize, CLWrapper *inputData, CLWrapper *selectors, CLWrapper *outputData ) {
    throw runtime_error("propagate not implemented for this child type");
}
VIRTUAL void PoolingPropagate::propagate( int batchSize, float *input, int *selectors, float *output ) {
//    cout << "PoolingPropagate::propagate( float * )" << endl;
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    CLWrapper *selectorsWrapper = cl->wrap( getResultsSize( batchSize ), selectors );
    CLWrapper *outputWrapper = cl->wrap( getResultsSize( batchSize ), output );

    inputWrapper->copyToDevice();
    propagate( batchSize, inputWrapper, selectorsWrapper, outputWrapper );
    selectorsWrapper->copyToHost();    
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete selectorsWrapper;
    delete inputWrapper;
}
VIRTUAL int PoolingPropagate::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int PoolingPropagate::getResultsSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}


