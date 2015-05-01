// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "stringhelper.h"
#include "DropoutForwardCpu.h"
#include "DropoutForwardGpuNaive.h"

#include "DropoutForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutForward::DropoutForward( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputImageSize( inputImageSize ),
        dropRatio( dropRatio ),
        outputImageSize( inputImageSize ) {
//    if( inputImageSize % dropoutSize != 0 ) {
//        throw runtime_error("inputImageSize should be an exact multiple of dropoutsize: " + toString( inputImageSize ) + " " + toString(dropoutSize ) );
//    }
}
STATIC DropoutForward *DropoutForward::instance( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutForwardGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
//    return new DropoutForwardCpu( cl, padZeros, numPlanes, inputImageSize, dropoutSize );
}
STATIC DropoutForward *DropoutForward::instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutForwardCpu( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutForward *DropoutForward::instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    if( idx == 0 ) {
        return new DropoutForwardCpu( cl, numPlanes, inputImageSize, dropRatio );
    }
    if( idx == 1 ) {
        return new DropoutForwardGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("DropoutForward::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void DropoutForward::forward( int batchSize, CLWrapper *masksWrapper, CLWrapper *inputData, CLWrapper *outputData ) {
    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void DropoutForward::forward( int batchSize, unsigned char *masks, float *input, float *output ) {
//    cout << "DropoutForward::forward( float * )" << endl;
    int inputLinearSize = getInputSize( batchSize );
    CLWrapper *masksWrapper = cl->wrap( inputLinearSize, masks );
    CLWrapper *inputWrapper = cl->wrap( inputLinearSize, input );
    CLWrapper *outputWrapper = cl->wrap( getOutputSize( batchSize ), output );

    masksWrapper->copyToDevice();
    inputWrapper->copyToDevice();
    forward( batchSize, masksWrapper, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
    delete masksWrapper;
}
VIRTUAL int DropoutForward::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int DropoutForward::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}


