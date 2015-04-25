// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "ActivationPropagateCpu.h"
#include "ActivationPropagateGpuNaive.h"

#include "ActivationPropagate.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationPropagate::ActivationPropagate( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        outputImageSize( inputImageSize ),
        fn(fn) {
}
STATIC ActivationPropagate *ActivationPropagate::instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    return new ActivationPropagateGpuNaive( cl, numPlanes, inputImageSize, fn );
//    return new ActivationPropagateCpu( cl, numPlanes, inputImageSize );
}
STATIC ActivationPropagate *ActivationPropagate::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    return new ActivationPropagateGpuNaive( cl, numPlanes, inputImageSize, fn );
}
STATIC ActivationPropagate *ActivationPropagate::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    if( idx == 0 ) {
        return new ActivationPropagateCpu( cl, numPlanes, inputImageSize, fn );
    }
    if( idx == 1 ) {
        return new ActivationPropagateGpuNaive( cl, numPlanes, inputImageSize, fn );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("ActivationPropagate::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void ActivationPropagate::propagate( int batchSize, CLWrapper *inputData, CLWrapper *outputData ) {
    throw runtime_error("propagate not implemented for this child type");
}
VIRTUAL void ActivationPropagate::propagate( int batchSize, float *input, float *output ) {
//    cout << "ActivationPropagate::propagate( float * )" << endl;
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    CLWrapper *outputWrapper = cl->wrap( getResultsSize( batchSize ), output );

    inputWrapper->copyToDevice();
    propagate( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
}
VIRTUAL int ActivationPropagate::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int ActivationPropagate::getResultsSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}

