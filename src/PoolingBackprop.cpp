// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

#include "PoolingBackpropCpu.h"
#include "PoolingBackpropGpuNaive.h"

#include "PoolingBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC PoolingBackprop *PoolingBackprop::instance( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    return new PoolingBackpropGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceForTest( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize) {
    return new PoolingBackpropCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
STATIC PoolingBackprop *PoolingBackprop::instanceSpecific( int idx, OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingBackpropCpu( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    if( idx == 1 ) {
        return new PoolingBackpropGpuNaive( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    }
    throw runtime_error("PoolingBackprop::instanceSpecific, idx not known: " + toString( idx ) );
}
PoolingBackprop::PoolingBackprop( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputImageSize, int poolingSize ) :
        cl( cl ),
        padZeros( padZeros ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        poolingSize( poolingSize ),
//        poolingSizeSquared( poolingSize * poolingSize ),
        outputImageSize( padZeros ? ( inputImageSize + poolingSize - 1 ) / poolingSize : inputImageSize / poolingSize ) {
//    if( inputImageSize % poolingSize != 0 ) {
//        throw runtime_error("inputImageSize should be an exact multiple of poolingsize: " + toString( inputImageSize ) + " " + toString(poolingSize ) );
//    }
}
VIRTUAL int PoolingBackprop::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int PoolingBackprop::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL void PoolingBackprop::backward( int batchSize, float *gradOutput, int *selectors, float *gradInput ) {
//    cout << "PoolingBackprop::backward( float * )" << endl;
    StatefulTimer::instance()->timeCheck("PoolingBackprop::backward float->wrapper start" );
    CLWrapper *gradOutputWrapper = cl->wrap( getOutputSize(batchSize), gradOutput );
    CLWrapper *selectorsWrapper = cl->wrap( getOutputSize(batchSize), selectors );
    CLWrapper *gradInputWrapper = cl->wrap( getInputSize(batchSize), gradInput );

    gradOutputWrapper->copyToDevice();
    selectorsWrapper->copyToDevice();

    backward( batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper );

    selectorsWrapper->copyToHost();
    gradInputWrapper->copyToHost();

    delete gradOutputWrapper;
    delete selectorsWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("PoolingBackprop::backward float->wrapper end" );
}
VIRTUAL void PoolingBackprop::backward( int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper, CLWrapper *gradInputWrapper ) {
    throw runtime_error("PoolingBackprop::backward wrappers not implemented" );
}

