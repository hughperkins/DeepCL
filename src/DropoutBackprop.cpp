// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

#include "DropoutBackpropCpu.h"
#include "DropoutBackpropGpuNaive.h"

#include "DropoutBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC DropoutBackprop *DropoutBackprop::instance( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    return new DropoutBackpropGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackprop *DropoutBackprop::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio) {
    return new DropoutBackpropGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
}
STATIC DropoutBackprop *DropoutBackprop::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) {
    if( idx == 0 ) {
        return new DropoutBackpropCpu( cl, numPlanes, inputImageSize, dropRatio );
    }
    if( idx == 1 ) {
        return new DropoutBackpropGpuNaive( cl, numPlanes, inputImageSize, dropRatio );
    }
    throw runtime_error("DropoutBackprop::instanceSpecific, idx not known: " + toString( idx ) );
}
DropoutBackprop::DropoutBackprop( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
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
VIRTUAL int DropoutBackprop::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputImageSize * inputImageSize;
}
VIRTUAL int DropoutBackprop::getOutputSize(int batchSize) {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL void DropoutBackprop::backward( int batchSize, uchar *mask, float *gradOutput, float *gradInput ) {
//    cout << "DropoutBackprop::backward( float * )" << endl;
    StatefulTimer::instance()->timeCheck("DropoutBackprop::backward float->wrapper start" );
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
    StatefulTimer::instance()->timeCheck("DropoutBackprop::backward float->wrapper end" );
}
VIRTUAL void DropoutBackprop::backward( int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper ) {
    throw runtime_error("DropoutBackprop::backward wrappers not implemented" );
}

