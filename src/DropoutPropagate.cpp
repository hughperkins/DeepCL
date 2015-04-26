// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "DropoutPropagateCpu.h"
#include "DropoutPropagateGpuNaive.h"

#include "DropoutPropagate.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutPropagate::DropoutPropagate( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        dropRatio( dropRatio ),
        outputImageSize( inputImageSize ) {
//    if( inputImageSize % dropoutSize != 0 ) {
//        throw runtime_error("inputImageSize should be an exact multiple of dropoutsize: " + toString( inputImageSize ) + " " + toString(dropoutSize ) );
//    }
}
STATIC DropoutPropagate *DropoutPropagate::instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutPropagateGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
//    return new DropoutPropagateCpu( cl, padZeros, numPlanes, inputImageSize, dropoutSize );
}
STATIC DropoutPropagate *DropoutPropagate::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutPropagateGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutPropagate *DropoutPropagate::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    if( idx == 0 ) {
        return new DropoutPropagateCpu( cl, numPlanes, inputImageSize, dropRatio );
    }
    if( idx == 1 ) {
        return new DropoutPropagateGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("DropoutPropagate::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void DropoutPropagate::propagate( int batchSize, CLWrapper *inputData, CLWrapper *outputData ) {
    throw runtime_error("propagate not implemented for this child type");
}
VIRTUAL void DropoutPropagate::propagate( int batchSize, float *input, float *output ) {
//    cout << "DropoutPropagate::propagate( float * )" << endl;
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    CLWrapper *outputWrapper = cl->wrap( getResultsSize( batchSize ), output );

    inputWrapper->copyToDevice();
    propagate( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
}
VIRTUAL int DropoutPropagate::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int DropoutPropagate::getResultsSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}


