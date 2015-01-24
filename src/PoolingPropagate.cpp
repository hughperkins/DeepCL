// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "PoolingPropagateCpu.h"

#include "PoolingPropagate.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingPropagate::PoolingPropagate( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) :
        cl( cl ),
        numPlanes( numPlanes ),
        inputBoardSize( inputBoardSize ),
        poolingSize( poolingSize ),
        outputBoardSize( inputBoardSize / poolingSize ) {
}
STATIC PoolingPropagate *PoolingPropagate::instance( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) {
    return new PoolingPropagateCpu( cl, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingPropagate *PoolingPropagate::instanceForTest( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) {
    return new PoolingPropagateCpu( cl, numPlanes, inputBoardSize, poolingSize );
}
STATIC PoolingPropagate *PoolingPropagate::instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) {
    if( idx == 0 ) {
        return new PoolingPropagateCpu( cl, numPlanes, inputBoardSize, poolingSize );
    }
    throw runtime_error("PoolingPropagate::instanceSpecific idx not known: " + toString( idx ) );
}
VIRTUAL void PoolingPropagate::propagate( int batchSize, CLWrapper *inputData, CLWrapper *outputData ) {
    throw runtime_error("propagate not implemented for this child type");
}
VIRTUAL float *PoolingPropagate::propagate( int batchSize, float *input ) {
    CLWrapper *inputWrapper = cl->wrap( getInputSize( batchSize ), input );
    float *output = new float[ getResultsSize( batchSize ) ];
    CLWrapper *outputWrapper = cl->wrap( getResultsSize( batchSize ), output );
    throw runtime_error("propagate not implemented for this child type");

    inputWrapper->copyToDevice();
    propagate( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete inputWrapper;
    return output;
}
VIRTUAL int PoolingPropagate::getInputSize( int batchSize ) {
    return batchSize * numPlanes * inputBoardSize * inputBoardSize;
}
VIRTUAL int PoolingPropagate::getResultsSize(int batchSize) {
    return batchSize * numPlanes * inputBoardSize * inputBoardSize / poolingSize / poolingSize;
}


