// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "StatefulTimer.h"
#include "SGD.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL SGD::~SGD() {
    delete lastUpdateWrapper;
    delete[] lastUpdate;
    //delete kernel;
}

VIRTUAL void SGD::setMomentum( float momentum ) {
    this->momentum = momentum;
}

VIRTUAL void SGD::updateWeights(CLWrapper *gradientsWrapper, CLWrapper *weightsWrapper ) {
    // first, determine updates, based on gradient, and last updates
    // can copy this directly into lastUpdateWrapper
    // then, we update the weights
    // actually, can directly update the weights too
    // so:
    // input: last updates, current gradient, current weights
    // 1. update last updates to this updates
    // 2. update weights
    // that's it :-)

    StatefulTimer::instance()->timeCheck("SGD::updateWeights start" );

    kernel  ->in( numWeights )
            ->in( learningRate )
            ->in( momentum )
            ->inout( lastUpdateWrapper )
            ->in( gradientsWrapper )
            ->inout( weightsWrapper );
    int globalSize = numWeights;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

    StatefulTimer::instance()->timeCheck("SGD::updateWeights end" );
}

SGD::SGD( OpenCLHelper *cl, int numWeights ) :
        cl( cl ),
        kernel( 0 ),
        numWeights( numWeights ),
        learningRate(1.0f),
        momentum( 0.0f )
    { // should we handle bias separately?  maybe... not?
      // or each layer could have one trainer for biases, and one for the
      // non-biases?  Maybe kind of ok?

    // lastUpdate buffer never needs to change size,
    //  since number of weights is invariant with batchSize etc
    lastUpdate = new float[numWeights];
    for( int i = 0; i < numWeights; i++ ) {
        lastUpdate[i] = 0.0f;
    }
    lastUpdateWrapper = cl->wrap( numWeights, lastUpdate );
    lastUpdateWrapper->copyToDevice();

    string options = "";

    static CLKernel *kernel = 0; // since kernel contains no defines, we 
                                 // can share it across all SGD instances,
                                 // save some compile time :-)
    if( kernel != 0 ) {
        this->kernel = kernel;
    }

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/SGD.cl", "updateWeights", 'options' )
    // ]]]
    // generated using cog, from cl/SGD.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void updateWeights(\n" 
    "        const int N,\n" 
    "        const float learningRate,\n" 
    "        const float momentum,\n" 
    "        global float *lastUpdate,\n" 
    "        global const float *currentGradients,\n" 
    "        global float *weights\n" 
    "            ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    // first update the update\n" 
    "    lastUpdate[globalId] =\n" 
    "        learningRate * currentGradients[globalId] +\n" 
    "        momentum * lastUpdate[globalId];\n" 
    "    // now update the weight\n" 
    "    weights[globalId] += lastUpdate[globalId];\n" 
    "    // thats it... :-)\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "updateWeights", options, "cl/SGD.cl" );
    // [[[end]]]
    this->kernel = kernel;
}


