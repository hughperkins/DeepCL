// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/StatefulTimer.h"
#include "EasyCL.h"
#include "clmath/GpuOp.h"
#include "templates/LuaTemplater.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

/// \brief calculates destinationWrapper += deltaWrapper
VIRTUAL void GpuOp::apply2_inplace(int N, CLWrapper*destinationWrapper, float scalar, Op2 *op) {
    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace start");

    string kernelName = "GpuOp::" + op->getName() + "_inplace_scalar";
    if(!cl->kernelExists(kernelName) ) {
        buildKernelScalar(kernelName, op, true);
    }
    CLKernel *kernel = cl->getKernel(kernelName);

    kernel->in(N);
    kernel->inout(destinationWrapper);
    kernel->in(scalar);
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace end");
}
VIRTUAL void GpuOp::apply2_inplace(int N, CLWrapper*destinationWrapper, CLWrapper *deltaWrapper, Op2 *op) {
    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace start");

    string kernelName = "GpuOp::" + op->getName() + "_inplace";
    if(!cl->kernelExists(kernelName) ) {
        buildKernel(kernelName, op, true);
    }
    CLKernel *kernel = cl->getKernel(kernelName);

    kernel->in(N);
    kernel->inout(destinationWrapper);
    kernel->in(deltaWrapper);
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace end");
}
VIRTUAL void GpuOp::apply2_outofplace(int N, CLWrapper*destinationWrapper, CLWrapper*one, CLWrapper *two, Op2 *op) {
    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace start");

    string kernelName = "GpuOp::" + op->getName() + "_outofplace";
    if(!cl->kernelExists(kernelName) ) {
        buildKernel(kernelName, op, false);
    }
    CLKernel *kernel = cl->getKernel(kernelName);

    kernel->in(N);
    kernel->inout(destinationWrapper);
    kernel->in(one);
    kernel->in(two);
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace end");
}
VIRTUAL void GpuOp::apply1_inplace(int N, CLWrapper*destinationWrapper, Op1 *op) {
    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace start");

    string kernelName = "GpuOp::" + op->getName() + "_inplace";
    if(!cl->kernelExists(kernelName) ) {
        buildKernel(kernelName, op, true);
    }
    CLKernel *kernel = cl->getKernel(kernelName);

    kernel->in(N);
    kernel->inout(destinationWrapper);
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace end");
}
VIRTUAL void GpuOp::apply1_outofplace(int N, CLWrapper*destinationWrapper, CLWrapper*one, Op1 *op) {
    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace start");

    string kernelName = "GpuOp::" + op->getName() + "_outofplace";
    if(!cl->kernelExists(kernelName) ) {
        buildKernel(kernelName, op, false);
    }
    CLKernel *kernel = cl->getKernel(kernelName);

    kernel->in(N);
    kernel->inout(destinationWrapper);
    kernel->in(one);
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("GpuOp::apply inplace end");
}
VIRTUAL GpuOp::~GpuOp() {
}
GpuOp::GpuOp(EasyCL *cl) :
        cl(cl) {
}
void GpuOp::buildKernel(std::string name, Op2 *op, bool inPlace) {

    // [[[cog
    // import stringify
    // stringify.write_kernel("kernel", "cl/per_element_op2.cl")
    // ]]]
    // generated using cog, from cl/per_element_op2.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static float operation(float val_one, float val_two) {\n"
    "    return {{operation}};\n"
    "}\n"
    "\n"
    "kernel void per_element_op2_inplace(const int N, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(target[globalId], source[globalId]);\n"
    "}\n"
    "\n"
    "kernel void per_element_op2_outofplace(const int N, global float *target, global float *one, global const float *two) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(one[globalId], two[globalId]);\n"
    "}\n"
    "\n"
    "";
    // [[[end]]]
    LuaTemplater templater;
    templater.set("operation", op->getOperationString());
    string renderedKernel = templater.render(kernelSource);
    // cout << "renderedKernel:" << endl;
    // cout << renderedKernel << endl;

    string clKernelName = "per_element_op2_outofplace";
    if(inPlace) {
        clKernelName = "per_element_op2_inplace";
    }
    kernel = cl->buildKernelFromString(renderedKernel, clKernelName, "", "cl/per_element_op2.cl");
    cl->storeKernel(name, kernel, true);
}
void GpuOp::buildKernel(std::string name, Op1 *op, bool inPlace) {

    // [[[cog
    // import stringify
    // stringify.write_kernel("kernel", "cl/per_element_op1.cl")
    // ]]]
    // generated using cog, from cl/per_element_op1.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static float operation(float val_one) {\n"
    "    return {{operation}};\n"
    "}\n"
    "\n"
    "kernel void per_element_op1_inplace(const int N, global float *target) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(target[globalId]);\n"
    "}\n"
    "\n"
    "kernel void per_element_op1_outofplace(const int N, global float *target, global float *one) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(one[globalId]);\n"
    "}\n"
    "\n"
    "";
    // [[[end]]]
    LuaTemplater templater;
    templater.set("operation", op->getOperationString());
    string renderedKernel = templater.render(kernelSource);
    // cout << "renderedKernel:" << endl;
    // cout << renderedKernel << endl;

    string clKernelName = "per_element_op1_outofplace";
    if(inPlace) {
        clKernelName = "per_element_op1_inplace";
    }
    kernel = cl->buildKernelFromString(renderedKernel, clKernelName, "", "cl/per_element_op1.cl");
    cl->storeKernel(name, kernel, true);
}
void GpuOp::buildKernelScalar(std::string name, Op2 *op, bool inPlace) {

    // [[[cog
    // import stringify
    // stringify.write_kernel("kernel", "cl/per_element_op2_scalar.cl")
    // ]]]
    // generated using cog, from cl/per_element_op2_scalar.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static float operation(float val_one, float val_two) {\n"
    "    return {{operation}};\n"
    "}\n"
    "\n"
    "kernel void per_element_op2_inplace(const int N, global float *target, const float scalar) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(target[globalId], scalar);\n"
    "}\n"
    "\n"
    "kernel void per_element_op2_outofplace(const int N, global float *target, global float *source, const float scalar) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] = operation(source[globalId], scalar);\n"
    "}\n"
    "\n"
    "";
    // [[[end]]]
    LuaTemplater templater;
    templater.set("operation", op->getOperationString());
    string renderedKernel = templater.render(kernelSource);
    // cout << "renderedKernel:" << endl;
    // cout << renderedKernel << endl;

    string clKernelName = "per_element_op2_outofplace";
    if(inPlace) {
        clKernelName = "per_element_op2_inplace";
    }
    kernel = cl->buildKernelFromString(renderedKernel, clKernelName, "", "cl/per_element_op2_scalar.cl");
    cl->storeKernel(name, kernel, true);
}

