// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "EasyCL.h"
#include "activate/ActivationBackward.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"
#include "activate/ActivationFunction.h"

#include "activate/ActivationBackwardGpuNaive.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

VIRTUAL ActivationBackwardGpuNaive::~ActivationBackwardGpuNaive() {
    delete kernel;
//    delete kMemset;
}
VIRTUAL void ActivationBackwardGpuNaive::backward(int batchSize, CLWrapper *inputWrapper,
         CLWrapper *gradOutputWrapper, 
        CLWrapper *gradInputWrapper) {

    StatefulTimer::instance()->timeCheck("ActivationBackwardGpuNaive::backward start");

    int globalSize = batchSize * numPlanes * inputSize * inputSize;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->in(batchSize * numPlanes * inputSize * inputSize)
          ->in(inputWrapper)
          ->in(gradOutputWrapper)
          ->out(gradInputWrapper);
    globalSize = batchSize * numPlanes * outputSize * outputSize;
    workgroupSize = 64;
    numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();

    StatefulTimer::instance()->timeCheck("ActivationBackwardGpuNaive::backward end");
}
ActivationBackwardGpuNaive::ActivationBackwardGpuNaive(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) :
        ActivationBackward(cl, numPlanes, inputSize, fn) {
//    std::string options = "-D " + fn->getDefineName();
    string options = "";
    options += " -D gNumPlanes=" + toString(numPlanes);
    options += " -D gInputSize=" + toString(inputSize);
    options += " -D gInputSizeSquared=" + toString(inputSize * inputSize);
    options += " -D gOutputSize=" + toString(outputSize);
    options += " -D gOutputSizeSquared=" + toString(outputSize * outputSize);
    options += string(" -D ") + fn->getDefineName();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/applyActivationDeriv.cl", "backward", 'options')
    // ]]]
    // generated using cog, from cl/applyActivationDeriv.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 201, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// expected defines:\n"
    "// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU]\n"
    "\n"
    "#ifdef TANH\n"
    "    #define ACTIVATION_DERIV(output) (1 - output * output)\n"
    "#elif defined SCALEDTANH\n"
    "    #define ACTIVATION_DERIV(output) (0.66667f * (1.7159f - 1 / 1.7159f * output * output) )\n"
    "#elif defined SIGMOID\n"
    "    #define ACTIVATION_DERIV(output) (output * (1 - output) )\n"
    "#elif defined RELU\n"
    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)\n"
    "#elif defined ELU\n"
    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : output + 1)\n"
    "#elif defined LINEAR\n"
    "    #define ACTIVATION_DERIV(output) (1.0f)\n"
    "#endif\n"
    "\n"
    "//#ifdef ACTIVATION_DERIV\n"
    "//void kernel applyActivationDeriv(\n"
    "//        const int N,\n"
    "//        global float *inout) {\n"
    "//    int globalId = get_global_id(0);\n"
    "//    inout[globalId] = ACTIVATION_DERIV(inout[globalId]);\n"
    "//}\n"
    "//#endif\n"
    "\n"
    "#ifdef ACTIVATION_DERIV\n"
    "void kernel applyActivationDeriv(\n"
    "        const int N,\n"
    "        global float *target, global const float *source) {\n"
    "    int globalId = get_global_id(0);\n"
    "    if (globalId < N) {\n"
    "        target[globalId] *= ACTIVATION_DERIV(source[globalId]);\n"
    "    }\n"
    "  //  target[globalId] *= source[globalId];\n"
    "}\n"
    "#endif\n"
    "\n"
    "#ifdef ACTIVATION_DERIV\n"
    "void kernel backward(\n"
    "        const int N,\n"
    "        global const float *inputs,\n"
    "        global const float *gradOutput,\n"
    "        global float *gradInput) {\n"
    "    int globalId = get_global_id(0);\n"
    "    if (globalId < N) {\n"
    "        gradInput[globalId] = ACTIVATION_DERIV(inputs[globalId]) * gradOutput[globalId];\n"
    "            // probably not ideal to have the output and input separate?\n"
    "    }\n"
    "  //  target[globalId] *= source[globalId];\n"
    "}\n"
    "#endif\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backward", options, "cl/applyActivationDeriv.cl");
    // [[[end]]]
}

