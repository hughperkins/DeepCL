// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "OpenCLHelper.h"
#include "DropoutBackprop.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

#include "DropoutBackpropGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL DropoutBackpropGpuNaive::~DropoutBackpropGpuNaive() {
    delete kernel;
//    delete kMemset;
}
VIRTUAL void DropoutBackpropGpuNaive::backpropErrors( 
            int batchSize, 
            CLWrapper *maskWrapper, 
            CLWrapper *errorsWrapper, 
            CLWrapper *errorsForUpstreamWrapper ) 
        {

    StatefulTimer::instance()->timeCheck("DropoutBackpropGpuNaive::backpropErrors start" );

    // first, memset errors to 0 ...
//    kMemset ->out( errorsForUpstreamWrapper )
//            ->in( 0.0f )
//            ->in( batchSize * numPlanes * inputImageSize * inputImageSize );
//    int globalSize = batchSize * numPlanes * inputImageSize * inputImageSize;
//    int workgroupSize = 64;
//    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
//    kMemset->run_1d( numWorkgroups * workgroupSize, workgroupSize );
//    cl->finish();

    kernel  ->in( batchSize * numPlanes * outputImageSize * outputImageSize )
            ->in( maskWrapper )
            ->in( errorsWrapper )
            ->out( errorsForUpstreamWrapper );
    int globalSize = batchSize * numPlanes * outputImageSize * outputImageSize;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();

    StatefulTimer::instance()->timeCheck("DropoutBackpropGpuNaive::backpropErrors end" );
}
DropoutBackpropGpuNaive::DropoutBackpropGpuNaive( OpenCLHelper *cl, int numPlanes, int inputImageSize, float dropRatio ) :
        DropoutBackprop( cl, numPlanes, inputImageSize, dropRatio ) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
    options += " -D gNumPlanes=" + toString( numPlanes );
    options += " -D gInputImageSize=" + toString( inputImageSize );
    options += " -D gInputImageSizeSquared=" + toString( inputImageSize * inputImageSize );
    options += " -D gOutputImageSize=" + toString( outputImageSize );
    options += " -D gOutputImageSizeSquared=" + toString( outputImageSize * outputImageSize );
//    float inverseDropRatio = 1.0f / dropRatio;
    string dropRatioString = toString( dropRatio );
    if( dropRatioString.find( "." ) == string::npos ) {
        dropRatioString += ".0f";
    } else {
        dropRatioString += "f";
    }
//    cout << "inverseDropRatioString " << inverseDropRatioString << endl;
    options += " -D gDropRatio=" + dropRatioString;

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/dropout.cl", "backpropNaive", 'options' )
    // # stringify.write_kernel2( "kMemset", "cl/memset.cl", "memset", '""' )
    // ]]]
    // generated using cog, from cl/dropout.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void propagateNaive(\n" 
    "        const int N,\n" 
    "        global const unsigned char *mask,\n" 
    "        global const float *input,\n" 
    "        global float *output ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;\n" 
    "}\n" 
    "\n" 
    "kernel void backpropNaive(\n" 
    "        const int N,\n" 
    "        global const unsigned char *mask,\n" 
    "        global const float *errors,\n" 
    "        global float *output) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    output[globalId] = mask[globalId] == 1 ? errors[globalId] : 0.0f;\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "backpropNaive", options, "cl/dropout.cl" );
    // [[[end]]]
}

