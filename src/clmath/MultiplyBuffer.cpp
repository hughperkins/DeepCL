// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "MultiplyBuffer.h"
#include "util/stringhelper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL void MultiplyBuffer::multiply(int N, float multiplier, CLWrapper *in, CLWrapper *out) {
        StatefulTimer::instance()->timeCheck("MultiplyBuffer::multiply start");

    kernel  ->in(N)
            ->in(multiplier)
            ->in(in)
            ->out(out);

    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("MultiplyBuffer::multiply end");
}

VIRTUAL MultiplyBuffer::~MultiplyBuffer() {
//    delete kernel;
}

//VIRTUAL std::string MultiplyBuffer::floatToFloatString(float value) {
//    string floatString = toString(value);
//    if(floatString.find(".") == string::npos) {
//        floatString += ".0";
//    }
//    floatString += "f";
//    return floatString;
//}

MultiplyBuffer::MultiplyBuffer(EasyCL *cl) :
        cl(cl) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
//    options += " -DgN=" + toString(N);
//    options += " -DgMultiplier=" + floatToFloatString(multiplier);

    std::string kernelName = "multiplyConstant";
    if(cl->kernelExists(kernelName) ) {
        this->kernel = cl->getKernel(kernelName);
        return;
    }

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/copy.cl", "multiplyConstant", 'options')
    // ]]]
    // generated using cog, from cl/copy.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// simply copies from one to the other...\n"
    "// there might be something built-in to opencl for this\n"
    "// anyway... :-)\n"
    "kernel void copy(\n"
    "        const int N,\n"
    "        global const float *in,\n"
    "        global float *out) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    out[globalId] = in[globalId];\n"
    "}\n"
    "\n"
    "kernel void copy_with_offset(\n"
    "        const int N,\n"
    "        global const float *in,\n"
    "        const int inoffset,\n"
    "        global float *out,\n"
    "        const int outoffset) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    out[globalId + outoffset] = in[globalId + inoffset];\n"
    "}\n"
    "\n"
    "kernel void multiplyConstant(\n"
    "        const int N,\n"
    "        const float multiplier,\n"
    "        global const float *in,\n"
    "        global float *out) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    out[globalId] = multiplier * in[globalId];\n"
    "}\n"
    "\n"
    "kernel void multiplyInplace(\n"
    "        const int N,\n"
    "        const float multiplier,\n"
    "        global float *data) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    data[globalId] *= multiplier;\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "multiplyConstant", options, "cl/copy.cl");
    // [[[end]]]
    cl->storeKernel(kernelName, kernel, true);
    this->kernel = kernel;
}

