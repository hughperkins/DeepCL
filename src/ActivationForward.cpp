// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "stringhelper.h"
#include "ActivationForwardCpu.h"
#include "ActivationForwardGpuNaive.h"

#include "ActivationForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationForward::ActivationForward( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        outputImageSize( inputImageSize ),
        fn(fn) {
}
STATIC ActivationForward *ActivationForward::instance( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    return new ActivationForwardGpuNaive( cl, numPlanes, inputImageSize, fn );
//    return new ActivationForwardCpu( cl, numPlanes, inputImageSize );
}
STATIC ActivationForward *ActivationForward::instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    return new ActivationForwardCpu( cl, numPlanes, inputImageSize, fn );
}
STATIC ActivationForward *ActivationForward::instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, ActivationFunction const*fn ) {
    if( idx == 0 ) {
        return new ActivationForwardCpu( cl, numPlanes, inputImageSize, fn );
    }
    if( idx == 1 ) {
        return new ActivationForwardGpuNaive( cl, numPlanes, inputImageSize, fn );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("ActivationForward::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void ActivationForward::forward( int batchSize, CLWrapper *inputData, CLWrapper *outputData ) {
    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void ActivationForward::forward( int batchSize, float *input, float *output ) {
//    cout << "ActivationForward::forward( float * )" << endl;
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    CLWrapper *outputWrapper = cl->wrap( getOutputSize( batchSize ), output );

    inputWrapper->copyToDevice();
    outputWrapper->createOnDevice();
    forward( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
}
VIRTUAL int ActivationForward::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int ActivationForward::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}

