// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "conv/Forward3.h"
#include "conv/AddBias.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Forward3::~Forward3() {
    delete kernel;
    delete addBias;
}
VIRTUAL void Forward3::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("Forward3::forward begin");
//    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
//    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
    kernel->localFloats(square(dim.inputSize) );
    kernel->localFloats(square(dim.filterSize) * dim.inputPlanes);

    int workgroupsize = std::max(32, square(dim.outputSize) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();

    StatefulTimer::timeCheck("Forward3::forward after kernel1");

    if(dim.biased) {
        addBias->forward(batchSize, dim.numFilters, dim.outputSize,
                          outputWrapper, biasWrapper);
    }
}
Forward3::Forward3(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
            {

    addBias = new AddBias(cl);

    if(square(dim.outputSize) > cl->getMaxWorkgroupSize()) {
        throw runtime_error("cannot use forward3, since outputimagesize * outputimagesize > maxworkgroupsize");
    }

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward3.cl", "forward_3_by_n_outplane", 'options')
    // # stringify.write_kernel2("repeatedAdd", "cl/per_element_add.cl", "repeated_add", 'options')
    // ]]]
    // generated using cog, from cl/forward3.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// concept: each workgroup handles convolving one input example with one filtercube\n"
    "// and writing out one single output plane\n"
    "//\n"
    "// workgroup id organized like: [imageid][outplane]\n"
    "// local id organized like: [outrow][outcol]\n"
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n"
    "// number workgroups = 32\n"
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n"
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n"
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n"
    "// output are organized like [imageid][filterid][row][col]\n"
    "void kernel forward_3_by_n_outplane(const int batchSize,\n"
    "      global const float *images, global const float *filters,\n"
    "    global float *output,\n"
    "    local float *_upstreamImage, local float *_filterCube) {\n"
    "    const int globalId = get_global_id(0);\n"
    "\n"
    "    const int workgroupId = get_group_id(0);\n"
    "    const int workgroupSize = get_local_size(0);\n"
    "    const int n = workgroupId / gNumFilters;\n"
    "    const int outPlane = workgroupId % gNumFilters;\n"
    "\n"
    "    const int localId = get_local_id(0);\n"
    "    const int outputRow = localId / gOutputSize;\n"
    "    const int outputCol = localId % gOutputSize;\n"
    "\n"
    "    const int minu = gPadZeros ? max(-gHalfFilterSize, -outputRow) : -gHalfFilterSize;\n"
    "    const int maxu = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow  - gEven) : gHalfFilterSize - gEven;\n"
    "    const int minv = gPadZeros ? max(-gHalfFilterSize, -outputCol) : - gHalfFilterSize;\n"
    "    const int maxv = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol - gEven) : gHalfFilterSize - gEven;\n"
    "\n"
    "    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;\n"
    "\n"
    "    const int filterCubeLength = gInputPlanes * gFilterSizeSquared;\n"
    "    const int filterCubeGlobalOffset = outPlane * filterCubeLength;\n"
    "    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;\n"
    "    for (int i = 0; i < numPixelsPerThread; i++) {\n"
    "        int thisOffset = localId + i * workgroupSize;\n"
    "        if (thisOffset < filterCubeLength) {\n"
    "            _filterCube[thisOffset] = filters[filterCubeGlobalOffset + thisOffset];\n"
    "        }\n"
    "    }\n"
    "    // dont need a barrier, since we'll just run behind the barrier from the upstream image download\n"
    "\n"
    "    float sum = 0;\n"
    "    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {\n"
    "        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int i = 0; i < numUpstreamsPerThread; i++) {\n"
    "            int thisOffset = workgroupSize * i + localId;\n"
    "            if (thisOffset < gInputSizeSquared) {\n"
    "                _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        int filterImageOffset = upstreamPlane * gFilterSizeSquared;\n"
    "        for (int u = minu; u <= maxu; u++) {\n"
    "            int inputRow = outputRow + u;\n"
    "            #if gPadZeros == 0\n"
    "                inputRow += gHalfFilterSize;\n"
    "            #endif\n"
    "            int inputimagerowoffset = inputRow * gInputSize;\n"
    "            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    "            for (int v = minv; v <= maxv; v++) {\n"
    "                int inputCol = outputCol + v;\n"
    "                #if gPadZeros == 0\n"
    "                    inputCol += gHalfFilterSize;\n"
    "                #endif\n"
    "                if (localId < gOutputSizeSquared) {\n"
    "                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    // output are organized like [imageid][filterid][row][col]\n"
    "    int resultIndex = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId;\n"
    "    if (localId < gOutputSizeSquared) {\n"
    "        output[resultIndex ] = sum;\n"
    "    }\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "forward_3_by_n_outplane", options, "cl/forward3.cl");
    // [[[end]]]
}

