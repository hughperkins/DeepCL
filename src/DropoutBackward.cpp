// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "EasyCL.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

#include "DropoutBackwardCpu.h"
#include "DropoutBackwardGpuNaive.h"

#include "DropoutBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC DropoutBackward *DropoutBackward::instance( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutBackwardGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackward *DropoutBackward::instanceForTest( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio) {
    return new DropoutBackwardGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackward *DropoutBackward::instanceSpecific( int idx, EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    if( idx == 0 ) {
        return new DropoutBackwardCpu( cl, numPlanes, inputImageSize, dropRatio );
    }
    if( idx == 1 ) {
        return new DropoutBackwardGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
    }
    throw runtime_error("DropoutBackward::instanceSpecific, idx not known: " + toString( idx ) );
}
DropoutBackward::DropoutBackward( EasyCL *cl, int numPlanes, int inputImageSize, float dropRatio ) :
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
VIRTUAL int DropoutBackward::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int DropoutBackward::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL void DropoutBackward::backward( int batchSize, uchar *mask, float *gradOutput, float *gradInput ) {
//    cout << "DropoutBackward::backward( float * )" << endl;
    StatefulTimer::instance()->timeCheck("DropoutBackward::backward float->wrapper start" );
    CLWrapper *maskWrapper = cl->wrap( getOutputSize(batchSize), mask );
    CLWrapper *gradOutputWrapper = cl->wrap( getOutputSize(batchSize), gradOutput );
    CLWrapper *gradInputWrapper = cl->wrap( getInputSize(batchSize), gradInput );

    maskWrapper->copyToDevice();
    gradOutputWrapper->copyToDevice();
    gradInputWrapper->createOnDevice();

    backward( batchSize, maskWrapper, gradOutputWrapper, gradInputWrapper );

    gradInputWrapper->copyToHost();

    delete maskWrapper;
    delete gradOutputWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("DropoutBackward::backward float->wrapper end" );
}
VIRTUAL void DropoutBackward::backward( int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper ) {
    throw runtime_error("DropoutBackward::backward wrappers not implemented" );
}

