// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"

#include "StatefulTimer.h"
#include "stringhelper.h"

#include "PoolingPropagateGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingPropagateGpuNaive::PoolingPropagateGpuNaive( OpenCLHelper *cl, bool padZeros, int numPlanes, int inputBoardSize, int poolingSize ) :
        PoolingPropagate( cl, padZeros, numPlanes, inputBoardSize, poolingSize ) {
    string options = "";
    options += " -DgOutputBoardSize=" + toString( outputBoardSize );
    options += " -DgOutputBoardSizeSquared=" + toString( outputBoardSize * outputBoardSize );
    options += " -DgInputBoardSize=" + toString( inputBoardSize );
    options += " -DgInputBoardSizeSquared=" + toString( inputBoardSize * inputBoardSize );
    options += " -DgPoolingSize=" + toString( poolingSize );
    options += " -DgNumPlanes=" + toString( numPlanes );
    kernel = cl->buildKernel( "pooling.cl", "propagateNaive", options );
}
VIRTUAL PoolingPropagateGpuNaive::~PoolingPropagateGpuNaive() {
    delete kernel;
}
VIRTUAL void PoolingPropagateGpuNaive::propagate( int batchSize, CLWrapper *inputWrapper, CLWrapper *selectorsWrapper, CLWrapper *outputWrapper ) {
    StatefulTimer::instance()->timeCheck("PoolingPropagateGpuNaive::propagate start" );

    kernel->input( batchSize )->input( inputWrapper )->output( selectorsWrapper )->output( outputWrapper );
    int globalSize = batchSize * numPlanes * outputBoardSize * outputBoardSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("PoolingPropagateGpuNaive::propagate end" );
}

