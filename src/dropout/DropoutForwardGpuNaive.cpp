// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "EasyCL.h"

#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

#include "DropoutForwardGpuNaive.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL DropoutForwardGpuNaive::~DropoutForwardGpuNaive() {
    delete kernel;
}
VIRTUAL void DropoutForwardGpuNaive::forward(int batchSize, CLWrapper *masksWrapper, CLWrapper *inputWrapper, CLWrapper *outputWrapper) {
//    cout << StatefulTimer::instance()->prefix << "DropoutForwardGpuNaive::forward(CLWrapper *)" << endl;
    StatefulTimer::instance()->timeCheck("DropoutForwardGpuNaive::forward start");

    kernel  ->input(batchSize * numPlanes * outputSize * outputSize)
            ->input(masksWrapper)
            ->input(inputWrapper)
            ->output(outputWrapper);
    int globalSize = batchSize * numPlanes * outputSize * outputSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
//    cout << "DropoutForwardGpuNaive::forward batchsize=" << batchSize << " g=" << globalSize << " w=" << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

//    cout << "DropoutForwardGpuNaive::forward selectorswrapper:" << endl;
//    PrintBuffer::printInts(cl, selectorsWrapper, outputSize, outputSize);

    StatefulTimer::instance()->timeCheck("DropoutForwardGpuNaive::forward end");
}
DropoutForwardGpuNaive::DropoutForwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        DropoutForward(cl, numPlanes, inputSize, dropRatio) {
    string options = "";
    options += " -DgOutputSize=" + toString(outputSize);
    options += " -DgOutputSizeSquared=" + toString(outputSize * outputSize);
    options += " -DgInputSize=" + toString(inputSize);
    options += " -DgInputSizeSquared=" + toString(inputSize * inputSize);
    options += " -DgNumPlanes=" + toString(numPlanes);
//    float inverseDropRatio = 1.0f / dropRatio;
//    string inverseDropRatioString = toString(inverseDropRatio);
//    if(inverseDropRatioString.find(".") == string::npos) {
//        inverseDropRatioString += ".0f";
//    } else {
//        inverseDropRatioString += "f";
//    }
////    cout << "inverseDropRatioString " << inverseDropRatioString << endl;
//    options += " -D gInverseDropRatio=" + inverseDropRatioString;

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/dropout.cl", "forwardNaive", 'options')
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
    kernel = cl->buildKernelFromString(kernelSource, "forwardNaive", options, "cl/dropout.cl");
    // [[[end]]]
//    kernel = cl->buildKernel("dropout.cl", "forwardNaive", options);
}

