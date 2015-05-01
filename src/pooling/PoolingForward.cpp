// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/stringhelper.h"
#include "PoolingForwardCpu.h"
#include "PoolingForwardGpuNaive.h"

#include "PoolingForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingForward::PoolingForward( EasyCL *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) :
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
STATIC PoolingForward *PoolingForward::instance( EasyCL *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    return new PoolingForwardGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
//    return new PoolingForwardCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingForward *PoolingForward::instanceForTest( EasyCL *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    return new PoolingForwardGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingForward *PoolingForward::instanceSpecific( int idx, EasyCL *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingForwardCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    if( idx == 1 ) {
        return new PoolingForwardGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("PoolingForward::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void PoolingForward::forward( int batchSize, CLWrapper *inputData, CLWrapper *selectors, CLWrapper *outputData ) {
    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void PoolingForward::forward( int batchSize, float *input, int *selectors, float *output ) {
//    cout << "PoolingForward::forward( float * )" << endl;
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    CLWrapper *selectorsWrapper = cl->wrap( getOutputSize( batchSize ), selectors );
    CLWrapper *outputWrapper = cl->wrap( getOutputSize( batchSize ), output );

    inputWrapper->copyToDevice();
    forward( batchSize, inputWrapper, selectorsWrapper, outputWrapper );
    selectorsWrapper->copyToHost();    
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete selectorsWrapper;
    delete inputWrapper;
}
VIRTUAL int PoolingForward::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int PoolingForward::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}


