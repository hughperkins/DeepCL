#include "util/StatefulTimer.h"

#include "BackwardGpuNaive.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackwardGpuNaive::~BackwardGpuNaive() {
    delete kernel;
//    delete broadcastMultiply;
//    delete applyActivationDeriv;
}
VIRTUAL void BackwardGpuNaive::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("BackwardGpuNaive start");

    kernel
       ->in(batchSize)
        ->in(gradOutputWrapper)
       ->in(weightsWrapper)
        ->out(gradInputWrapper);

    int globalSize = batchSize * dim.inputCubeSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();
    StatefulTimer::instance()->timeCheck("BackwardGpuNaive after first kernel");

//    applyActivationDeriv->in(batchSize * dim.inputCubeSize)->in(gradInputWrapper)->in(inputDataWrapper);
//    applyActivationDeriv->run_1d(globalSize, workgroupsize);
//    cl->finish();
//    StatefulTimer::instance()->timeCheck("BackwardGpuNaive after applyActivationDeriv");
    
    StatefulTimer::instance()->timeCheck("BackwardGpuNaive end");
}
BackwardGpuNaive::BackwardGpuNaive(EasyCL *cl, LayerDimensions dim) :
        Backward(cl, dim)
            {
    std::string options = dim.buildOptionsString();
    options += ""; // " -D " + upstreamFn->getDefineName();
    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/backward.cl", "calcGradInput", 'options')
    // # stringify.write_kernel2("broadcastMultiply", "cl/backproperrorsv2.cl", "broadcast_multiply", 'options')
    // # stringify.write_kernel2("applyActivationDeriv", "cl/applyActivationDeriv.cl", "applyActivationDeriv", 'options')
    // # stringify.write_kernel("kernelSource", "ClConvolve.cl")
    // ]]]
    // generated using cog, from cl/backward.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// expected defines:\n"
    "//  - none\n"
    "\n"
    "// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]\n"
    "// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    "// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    "// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB\n"
    "void kernel calcGradInput(\n"
    "        const int batchSize,\n"
    "        global const float *gradOutput, global float *weights, global float *gradInput) {\n"
    "    int globalId = get_global_id(0);\n"
    "\n"
    "    const int upstreamImage2dId = globalId / gInputSizeSquared;\n"
    "\n"
    "    const int intraImageOffset = globalId % gInputSizeSquared;\n"
    "    const int upstreamRow = intraImageOffset / gInputSize;\n"
    "    const int upstreamCol = intraImageOffset % gInputSize;\n"
    "\n"
    "    const int upstreamPlane = upstreamImage2dId % gInputPlanes;\n"
    "    const int n = upstreamImage2dId / gInputPlanes;\n"
    "\n"
    "    if (n >= batchSize) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));\n"
    "    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);\n"
    "    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));\n"
    "    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);\n"
    "\n"
    "    float sumWeightTimesOutError = 0;\n"
    "    // aggregate over [outPlane][outRow][outCol]\n"
    "    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {\n"
    "        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
    "            int outRow = upstreamRow + gMargin - filterRow;\n"
    "            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
    "                int outCol = upstreamCol + gMargin - filterCol;\n"
    "                int resultIndex = (( n * gNumFilters\n"
    "                          + outPlane) * gOutputSize\n"
    "                          + outRow) * gOutputSize\n"
    "                          + outCol;\n"
    "                float thisError = gradOutput[resultIndex];\n"
    "                int thisWeightIndex = (( outPlane * gInputPlanes\n"
    "                                    + upstreamPlane) * gFilterSize\n"
    "                                    + filterRow) * gFilterSize\n"
    "                                    + filterCol;\n"
    "                float thisWeight = weights[thisWeightIndex];\n"
    "                float thisWeightTimesError = thisWeight * thisError;\n"
    "                sumWeightTimesOutError += thisWeightTimesError;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    gradInput[globalId] = sumWeightTimesOutError;\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "calcGradInput", options, "cl/backward.cl");
    // [[[end]]]
//    kernel = cl->buildKernel("backproperrorsv2.cl", "calcGradInput", options);
//    kernel = cl->buildKernelFromString(kernelSource, "calcGradInput", options);
}

