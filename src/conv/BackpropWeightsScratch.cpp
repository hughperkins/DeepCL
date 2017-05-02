// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackpropWeightsScratch.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsScratch::~BackpropWeightsScratch() {
    delete kernel;
}
VIRTUAL void BackpropWeightsScratch::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsScratch start");

    int workgroupsize = std::max(32, square(dim.filterSize) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;

    int localMemRequiredKB = (square(dim.outputSize) * 4 + square(dim.inputSize) * 4) / 1024 + 1;
    if(localMemRequiredKB >= cl->getLocalMemorySizeKB()) {
        throw runtime_error("local memory too small to use this kernel on this device.  Need: " + 
            toString(localMemRequiredKB) + "KB, but only have: " + 
            toString(cl->getLocalMemorySizeKB()) + "KB local memory");
    }

    const float learningMultiplier = learningRateToMultiplier(batchSize);

    kernel
       ->in(learningMultiplier)
       ->in(batchSize)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper)
       ->inout(gradWeightsWrapper);
    if(dim.biased) {
        kernel->inout(gradBiasWrapper);
    }
    kernel
        ->localFloats(square(dim.outputSize) )
        ->localFloats(square(dim.inputSize) );

    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeightsScratch end");
}
BackpropWeightsScratch::BackpropWeightsScratch(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    if(square(dim.filterSize) > cl->getMaxWorkgroupSize()) {
        throw runtime_error("cannot use BackpropWeightsScratch, since filterSize * filterSize > maxworkgroupsize");
    }

    std::string options = dim.buildOptionsString();
    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/BackpropWeightsScratch.cl", "backprop_floats_withscratch_dobias", 'options')
    // ]]]
    // generated using cog, from cl/BackpropWeightsScratch.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// expected defines:\n"
    "// BIASED (or not)\n"
    "\n"
    "// including cl/copyLocal.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static void copyLocal(local float *target, global float const *source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "static void copyGlobal(global float *target, local float const *source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "\n"
    "// including cl/ids.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "#define globalId (get_global_id(0))\n"
    "#define localId (get_local_id(0)  )\n"
    "#define workgroupId (get_group_id(0))\n"
    "#define workgroupSize (get_local_size(0))\n"
    "\n"
    "\n"
    "\n"
    "\n"
    "// workgroupId: [outputPlane][inputPlane]\n"
    "// localId: [filterRow][filterCol]\n"
    "// per-thread iteration: [n][outputRow][outputCol]\n"
    "// local: errorimage: outputSize * outputSize\n"
    "//        imageimage: inputSize * inputSize\n"
    "void kernel backprop_floats_withscratch_dobias(\n"
    "        const float learningRateMultiplier, const int batchSize,\n"
    "         global const float *gradOutput, global const float *images,\n"
    "        global float *gradWeights,\n"
    "        #ifdef BIASED\n"
    "             global float *gradBiasWeights,\n"
    "        #endif\n"
    "        local float *_errorImage, local float *_imageImage\n"
    " ) {\n"
    "    const int filterRow = localId / gFilterSize;\n"
    "    const int filterCol = localId % gFilterSize;\n"
    "\n"
    "    #define outPlane (workgroupId / gInputPlanes)\n"
    "    #define upstreamPlane (workgroupId % gInputPlanes)\n"
    "\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "    float thiswchange = 0;\n"
    "#ifdef BIASED\n"
    "    float thisbiaschange = 0;\n"
    "#endif\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        copyLocal(_imageImage, images + (n * gInputPlanes + upstreamPlane) * gInputSizeSquared, gInputSizeSquared);\n"
    "        copyLocal(_errorImage, gradOutput + (n * gNumFilters + outPlane) * gOutputSizeSquared, gOutputSizeSquared);\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (localId < gFilterSizeSquared) {\n"
    "            for (int outRow = 0; outRow < gOutputSize; outRow++) {\n"
    "                int upstreamRow = outRow - gMargin + filterRow;\n"
    "                for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
    "                    const int upstreamCol = outCol - gMargin + filterCol;\n"
    "                    #define proceed (upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize && upstreamCol < gInputSize)\n"
    "                    if (proceed) {\n"
    "                        // these defines reduce register pressure, compared to const\n"
    "                        // giving a 40% speedup on nvidia :-)\n"
    "                        #define resultIndex (outRow * gOutputSize + outCol)\n"
    "                        #define error (_errorImage[resultIndex])\n"
    "                        //const float error = _errorImage[resultIndex];\n"
    "                        #define upstreamDataIndex (upstreamRow * gInputSize + upstreamCol)\n"
    "                        #define upstreamResult (_imageImage[upstreamDataIndex])\n"
    "                        thiswchange += upstreamResult * error;\n"
    "    #ifdef BIASED\n"
    "                        thisbiaschange += error;\n"
    "    #endif\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    if (localId < gFilterSizeSquared) {\n"
    "        gradWeights[ workgroupId * gFilterSizeSquared + localId ] = learningRateMultiplier * thiswchange;\n"
    "    }\n"
    "#ifdef BIASED\n"
    "    #define writeBias (upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin)\n"
    "    if (writeBias) {\n"
    "        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
    "    }\n"
    "#endif\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backprop_floats_withscratch_dobias", options, "cl/BackpropWeightsScratch.cl");
    // [[[end]]]
//    kernel = cl->buildKernel("backpropgradWeights2.cl", "backprop_floats_withscratch_dobias", options);
//    kernel = cl->buildKernelFromString(kernelSource, "calcGradInput", options);
}

