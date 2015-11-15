// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "EasyCL.h"

#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

#include "PoolingForwardGpuNaive.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL PoolingForwardGpuNaive::~PoolingForwardGpuNaive() {
    delete kernel;
}
VIRTUAL void PoolingForwardGpuNaive::forward(int batchSize, CLWrapper *inputWrapper, CLWrapper *selectorsWrapper, CLWrapper *outputWrapper) {
//    cout << StatefulTimer::instance()->prefix << "PoolingForwardGpuNaive::forward(CLWrapper *)" << endl;
    StatefulTimer::instance()->timeCheck("PoolingForwardGpuNaive::forward start");

    kernel->input(batchSize)->input(inputWrapper)->output(selectorsWrapper)->output(outputWrapper);
    int globalSize = batchSize * numPlanes * outputSize * outputSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
//    cout << "PoolingForwardGpuNaive::forward batchsize=" << batchSize << " g=" << globalSize << " w=" << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

//    cout << "PoolingForwardGpuNaive::forward selectorswrapper:" << endl;
//    PrintBuffer::printInts(cl, selectorsWrapper, outputSize, outputSize);

    StatefulTimer::instance()->timeCheck("PoolingForwardGpuNaive::forward end");
}
PoolingForwardGpuNaive::PoolingForwardGpuNaive(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) :
        PoolingForward(cl, padZeros, numPlanes, inputSize, poolingSize) {
    string options = "";
    options += " -DgOutputSize=" + toString(outputSize);
    options += " -DgOutputSizeSquared=" + toString(outputSize * outputSize);
    options += " -DgInputSize=" + toString(inputSize);
    options += " -DgInputSizeSquared=" + toString(inputSize * inputSize);
    options += " -DgPoolingSize=" + toString(poolingSize);
    options += " -DgNumPlanes=" + toString(numPlanes);

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/pooling.cl", "forwardNaive", 'options')
    // ]]]
    // generated using cog, from cl/pooling.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// every plane is independent\n"
    "// every example is independent\n"
    "// so, globalid can be: [n][plane][outputRow][outputCol]\n"
    "kernel void forwardNaive(const int batchSize, global const float *input, global int *selectors, global float *output) {\n"
    "    const int globalId = get_global_id(0);\n"
    "\n"
    "    const int intraImageOffset = globalId % gOutputSizeSquared;\n"
    "    const int outputRow = intraImageOffset / gOutputSize;\n"
    "    const int outputCol = intraImageOffset % gOutputSize;\n"
    "\n"
    "    const int image2dIdx = globalId / gOutputSizeSquared;\n"
    "    const int plane = image2dIdx % gNumPlanes;\n"
    "    const int n = image2dIdx / gNumPlanes;\n"
    "\n"
    "    if (n >= batchSize) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    const int inputRow = outputRow * gPoolingSize;\n"
    "    const int inputCol = outputCol * gPoolingSize;\n"
    "    const int inputImageOffset = (n * gNumPlanes + plane) * gInputSizeSquared;\n"
    "    int selector = 0;\n"
    "    int poolInputOffset = inputImageOffset + inputRow * gInputSize + inputCol;\n"
    "    float maxValue = input[ poolInputOffset ];\n"
    "    for (int dRow = 0; dRow < gPoolingSize; dRow++) {\n"
    "        for (int dCol = 0; dCol < gPoolingSize; dCol++) {\n"
    "            bool process = (inputRow + dRow < gInputSize) && (inputCol + dCol < gInputSize);\n"
    "            if (process) {\n"
    "                float thisValue = input[ poolInputOffset + dRow * gInputSize + dCol ];\n"
    "                if (thisValue > maxValue) {\n"
    "                    maxValue = thisValue;\n"
    "                    selector = dRow * gPoolingSize + dCol;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    output[ globalId ] = maxValue;\n"
    "    selectors[ globalId ] = selector;\n"
    "//    selectors[globalId] = 123;\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "forwardNaive", options, "cl/pooling.cl");
    // [[[end]]]
//    kernel = cl->buildKernel("pooling.cl", "forwardNaive", options);
}

