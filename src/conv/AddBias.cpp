// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL AddBias::~AddBias() {
}
VIRTUAL void AddBias::forward(
        int batchSize, int numFilters, int outputSize,
        CLWrapper *outputWrapper,
        CLWrapper *biasWrapper
            ) {
    StatefulTimer::timeCheck("AddBias::forward begin");

    kernel->in(batchSize * numFilters * outputSize * outputSize)
        ->in(numFilters)
        ->in(outputSize * outputSize)
        ->inout(outputWrapper)->in(biasWrapper);
    int globalSize = batchSize * numFilters * outputSize * outputSize;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::timeCheck("AddBias::forward after repeatedAdd");
}
AddBias::AddBias(EasyCL *cl) :
        cl(cl)
            {
    string kernelName = "AddBias.per_element_add";
    if(cl->kernelExists(kernelName) ) {
        this->kernel = cl->getKernel(kernelName);
        return;
    }

    std::string options = "";

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/per_element_add.cl", "repeated_add", 'options')
    // ]]]
    // generated using cog, from cl/per_element_add.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "kernel void per_element_add(const int N, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[globalId];\n"
    "}\n"
    "\n"
    "// adds source to target\n"
    "// tiles source as necessary, according to tilingSize\n"
    "kernel void per_element_tiled_add(const int N, const int tilingSize, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[globalId % tilingSize];\n"
    "}\n"
    "\n"
    "kernel void repeated_add(const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[ (globalId / repeatSize) % sourceSize ];\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "repeated_add", options, "cl/per_element_add.cl");
    // [[[end]]]

    cl->storeKernel(kernelName, kernel, true);
}

