// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsNaive.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsNaive::~BackpropWeightsNaive() {
//    cout << "~backpropgradWeights2naive: deleting kernel" << endl;
    delete kernel;
}
VIRTUAL void BackpropWeightsNaive::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsNaive start");

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

    int globalSize = dim.filtersSize;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ((globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
    kernel->run_1d(globalSize, workgroupsize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeightsNaive end");
}
BackpropWeightsNaive::BackpropWeightsNaive(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    std::string options = dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/backpropweights.cl", "backprop_floats", 'options')
    // ]]]
    // generated using cog, from cl/backpropweights.cl:
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
    "// globalId: [outPlane][inputPlane][filterRow][filterCol]\n"
    "// per-thread iteration: [n][outputRow][outputCol]\n"
    "void kernel backprop_floats(const float learningRateMultiplier,\n"
    "        const int batchSize,\n"
    "         global const float *gradOutput, global const float *images,\n"
    "        global float *gradWeights\n"
    "        #ifdef BIASED\n"
    "            , global float *gradBiasWeights\n"
    "        #endif\n"
    " ) {\n"
    "    int globalId = get_global_id(0);\n"
    "    if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    int IntraFilterOffset = globalId % gFilterSizeSquared;\n"
    "    int filterRow = IntraFilterOffset / gFilterSize;\n"
    "    int filterCol = IntraFilterOffset % gFilterSize;\n"
    "\n"
    "    int filter2Id = globalId / gFilterSizeSquared;\n"
    "    int outPlane = filter2Id / gInputPlanes;\n"
    "    int upstreamPlane = filter2Id % gInputPlanes;\n"
    "\n"
    "    float thiswchange = 0;\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "#ifdef BIASED\n"
    "    float thisbiaschange = 0;\n"
    "#endif\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        for (int outRow = 0; outRow < gOutputSize; outRow++) {\n"
    "            int upstreamRow = outRow - gMargin + filterRow;\n"
    "            for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
    "                int upstreamCol = outCol - gMargin + filterCol;\n"
    "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize\n"
    "                    && upstreamCol < gInputSize;\n"
    "                if (proceed) {\n"
    "                    int resultIndex = (( n * gNumFilters\n"
    "                              + outPlane) * gOutputSize\n"
    "                              + outRow) * gOutputSize\n"
    "                              + outCol;\n"
    "                    float error = gradOutput[resultIndex];\n"
    "                    int upstreamDataIndex = (( n * gInputPlanes\n"
    "                                     + upstreamPlane) * gInputSize\n"
    "                                     + upstreamRow) * gInputSize\n"
    "                                     + upstreamCol;\n"
    "                    float upstreamResult = images[upstreamDataIndex];\n"
    "                    float thisimagethiswchange = upstreamResult * error;\n"
    "                    thiswchange += thisimagethiswchange;\n"
    "    #ifdef BIASED\n"
    "                    thisbiaschange += error;\n"
    "    #endif\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
    "#ifdef BIASED\n"
    "    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n"
    "    if (writeBias) {\n"
    "        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
    "    }\n"
    "#endif\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backprop_floats", options, "cl/backpropweights.cl");
    // [[[end]]]
}

