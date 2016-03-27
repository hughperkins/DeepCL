// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "MultiplyInPlace.h"
#include "util/stringhelper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

/// \brief calculates data *= multiplier
VIRTUAL void MultiplyInPlace::multiply(int N, float multiplier, CLWrapper *data) {
        StatefulTimer::instance()->timeCheck("MultiplyInPlace::multiply start");

    kernel  ->in(N)
            ->in(multiplier)
            ->inout(data);

    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("MultiplyInPlace::multiply end");
}
VIRTUAL MultiplyInPlace::~MultiplyInPlace() {
//    delete kernel;
}
MultiplyInPlace::MultiplyInPlace(EasyCL *cl) :
        cl(cl) {
    string options = "";

    std::string kernelName = "copy.multiplyInplace";
    if(cl->kernelExists(kernelName) ) {
        this->kernel = cl->getKernel(kernelName);
//        cout << "MultiplyInPlace kernel already built => reusing" << endl;
        return;
    }
    cout << "MultiplyInPlace: building kernel" << endl;

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/copy.cl", "multiplyInplace", 'options')
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
    kernel = cl->buildKernelFromString(kernelSource, "multiplyInplace", options, "cl/copy.cl");
    // [[[end]]]
    cl->storeKernel(kernelName, kernel, true);
    this->kernel = kernel;
}
// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "MultiplyInPlace.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


