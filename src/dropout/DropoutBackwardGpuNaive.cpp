// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "EasyCL.h"
#include "DropoutBackward.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

#include "DropoutBackwardGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL DropoutBackwardGpuNaive::~DropoutBackwardGpuNaive() {
    delete kernel;
//    delete kMemset;
}
VIRTUAL void DropoutBackwardGpuNaive::backward( 
            int batchSize, 
            CLWrapper *maskWrapper, 
            CLWrapper *gradOutputWrapper, 
            CLWrapper *gradInputWrapper) 
        {

    StatefulTimer::instance()->timeCheck("DropoutBackwardGpuNaive::backward start");

    // first, memset errors to 0 ...
//    kMemset ->out(gradInputWrapper)
//            ->in(0.0f)
//            ->in(batchSize * numPlanes * inputSize * inputSize);
//    int globalSize = batchSize * numPlanes * inputSize * inputSize;
//    int workgroupSize = 64;
//    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
//    kMemset->run_1d(numWorkgroups * workgroupSize, workgroupSize);
//    cl->finish();

    kernel  ->in(batchSize * numPlanes * outputSize * outputSize)
            ->in(maskWrapper)
            ->in(gradOutputWrapper)
            ->out(gradInputWrapper);
    int globalSize = batchSize * numPlanes * outputSize * outputSize;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("DropoutBackwardGpuNaive::backward end");
}
DropoutBackwardGpuNaive::DropoutBackwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        DropoutBackward(cl, numPlanes, inputSize, dropRatio) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
    options += " -D gNumPlanes=" + toString(numPlanes);
    options += " -D gInputSize=" + toString(inputSize);
    options += " -D gInputSizeSquared=" + toString(inputSize * inputSize);
    options += " -D gOutputSize=" + toString(outputSize);
    options += " -D gOutputSizeSquared=" + toString(outputSize * outputSize);
//    float inverseDropRatio = 1.0f / dropRatio;
    string dropRatioString = toString(dropRatio);
    if(dropRatioString.find(".") == string::npos) {
        dropRatioString += ".0f";
    } else {
        dropRatioString += "f";
    }
//    cout << "inverseDropRatioString " << inverseDropRatioString << endl;
    options += " -D gDropRatio=" + dropRatioString;

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/dropout.cl", "backpropNaive", 'options')
    // # stringify.write_kernel2("kMemset", "cl/memset.cl", "cl_memset", '""')
    // ]]]
    // generated using cog, from cl/dropout.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "kernel void forwardNaive(\n"
    "        const int N,\n"
    "        global const unsigned char *mask,\n"
    "        global const float *input,\n"
    "        global float *output) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;\n"
    "}\n"
    "\n"
    "kernel void backpropNaive(\n"
    "        const int N,\n"
    "        global const unsigned char *mask,\n"
    "        global const float *gradOutput,\n"
    "        global float *output) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    output[globalId] = mask[globalId] == 1 ? gradOutput[globalId] : 0.0f;\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backpropNaive", options, "cl/dropout.cl");
    // [[[end]]]
}

