// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/StatefulTimer.h"
#include "conv/ReduceSegments.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL ReduceSegments::~ReduceSegments() {
}
VIRTUAL void ReduceSegments::reduce(
        int totalLength,
        int segmentLength,
        CLWrapper *inputWrapper,
        CLWrapper *outputWrapper
            ) {
    StatefulTimer::timeCheck("ReduceSegments::reduce begin");

    if(totalLength % segmentLength != 0) {
        throw runtime_error("ReduceSegments: totalLength should be multiple of segmentLength");
    }
    const int numSegments = totalLength / segmentLength;
    kernel
        ->in(numSegments)
        ->in(segmentLength)
        ->in(inputWrapper)
        ->out(outputWrapper);
    int numWorkgroups = (numSegments + 64 - 1) / 64;
    kernel->run_1d(numWorkgroups * 64, 64);
    cl->finish();

    StatefulTimer::timeCheck("ReduceSegments::reduce end");
}
ReduceSegments::ReduceSegments(EasyCL *cl) :
        cl(cl)
            {
    string kernelName = "ReduceSegments.reduce_segments";
    if(cl->kernelExists(kernelName) ) {
        this->kernel = cl->getKernel(kernelName);
        return;
    }

    std::string options = "";

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/reduce_segments.cl", "reduce_segments", 'options')
    // ]]]
    // generated using cog, from cl/reduce_segments.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "kernel void reduce_segments(const int numSegments, const int segmentLength,\n"
    "        global float const *in, global float* out) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    const int segmentId = globalId;\n"
    "\n"
    "    if (segmentId >= numSegments) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    float sum = 0;\n"
    "    global const float *segment = in + segmentId * segmentLength;\n"
    "    for (int i = 0; i < segmentLength; i++) {\n"
    "        sum += segment[i];\n"
    "    }\n"
    "    out[segmentId] = sum;\n"
    "}\n"
    "\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "reduce_segments", options, "cl/reduce_segments.cl");
    // [[[end]]]

    cl->storeKernel(kernelName, kernel, true);
}

