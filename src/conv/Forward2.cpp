// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "conv/Forward2.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL Forward2::~Forward2() {
    delete kernel;
    delete addBias;
}
// only works for small filters
// condition: square(dim.filterSize) * dim.inputPlanes * 4 < 5000 (about 5KB)
VIRTUAL void Forward2::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("Forward2::forward START");
    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
//        cout << "square(outputSize) " << square(outputSize) << endl;
    kernel->localFloats(square(dim.inputSize) );
    kernel->localFloats(square(dim.filterSize) * dim.inputPlanes);
//    cout << "forward2 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();
    StatefulTimer::timeCheck("Forward2::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward2::forward END");
}
Forward2::Forward2(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
    if(square(dim.outputSize) > cl->getMaxWorkgroupSize()) {
        throw runtime_error("cannot use forward2, since outputimagesize * outputimagesize > maxworkgroupsize");
    }

    addBias = new AddBias(cl);

    this->workgroupSize = square(dim.outputSize);
    // round up to nearest 32, so dont waste threads:
    this->workgroupSize = (( workgroupSize + 32 - 1) / 32) * 32;
    this->numWorkgroups = dim.numFilters;
    this->globalSize = this->workgroupSize * this->numWorkgroups;

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();
    options += " -DgWorkgroupSize=" + toString(this->workgroupSize);
    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward2.cl", "forward_2_by_outplane", 'options')
    // ]]]
    // generated using cog, from cl/forward2.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "void copyLocal(local float *target, global float const *source, const int N) {\n"
    "    int numLoops = (N + gWorkgroupSize - 1) / gWorkgroupSize;\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * gWorkgroupSize + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "#ifdef gOutputSize // for previous tests that dont define it\n"
    "// workgroup id organized like: [outplane]\n"
    "// local id organized like: [outrow][outcol]\n"
    "// each thread iterates over: [imageid][upstreamplane][filterrow][filtercol]\n"
    "// number workgroups = 32\n"
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n"
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n"
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n"
    "// output are organized like [imageid][filterid][row][col]\n"
    "// assumes filter is small, so filtersize * filterSize * inputPlanes * 4 < about 3KB\n"
    "//                            eg 5 * 5 * 32 * 4 = 3.2KB => ok :-)\n"
    "//                           but 28 * 28 * 32 * 4 = 100KB => less good :-P\n"
    "void kernel forward_2_by_outplane(\n"
    "        const int batchSize,\n"
    "        global const float *images, global const float *filters,\n"
    "        global float *output,\n"
    "        local float *_inputPlane, local float *_filterCube) {\n"
    "    const int globalId = get_global_id(0);\n"
    "\n"
    "    const int workgroupId = get_group_id(0);\n"
    "    const int workgroupSize = get_local_size(0);\n"
    "    const int outPlane = workgroupId;\n"
    "\n"
    "    const int localId = get_local_id(0);\n"
    "    const int outputRow = localId / gOutputSize;\n"
    "    const int outputCol = localId % gOutputSize;\n"
    "\n"
    "    #if gPadZeros == 1\n"
    "        const int minu = max(-gHalfFilterSize, -outputRow);\n"
    "        const int maxu = min(gHalfFilterSize, gOutputSize - 1 - outputRow) - gEven;\n"
    "        const int minv = max(-gHalfFilterSize, -outputCol);\n"
    "        const int maxv = min(gHalfFilterSize, gOutputSize - 1 - outputCol) - gEven;\n"
    "    #else\n"
    "        const int minu = -gHalfFilterSize;\n"
    "        const int maxu = gHalfFilterSize - gEven;\n"
    "        const int minv = -gHalfFilterSize;\n"
    "        const int maxv = gHalfFilterSize - gEven;\n"
    "    #endif\n"
    "\n"
    "    {\n"
    "        const int filterCubeLength = gInputPlanes * gFilterSizeSquared;\n"
    "        copyLocal(_filterCube,\n"
    "                filters + outPlane * filterCubeLength,\n"
    "                filterCubeLength);\n"
    "    }\n"
    "    // dont need a barrier, since we'll just run behind the barrier from the upstream image download\n"
    "\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        float sum = 0;\n"
    "        for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            copyLocal(_inputPlane,\n"
    "                       images + (n * gInputPlanes + upstreamPlane) * gInputSizeSquared,\n"
    "                       gInputSizeSquared);\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            int filterImageOffset = upstreamPlane * gFilterSizeSquared;\n"
    "            if (localId < gOutputSizeSquared) {\n"
    "                for (int u = minu; u <= maxu; u++) {\n"
    "                    int inputRow = outputRow + u;\n"
    "                    #if gPadZeros == 0\n"
    "                         inputRow += gHalfFilterSize;\n"
    "                    #endif\n"
    "                    int inputimagerowoffset = inputRow * gInputSize;\n"
    "                    int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    "                    for (int v = minv; v <= maxv; v++) {\n"
    "                        int inputCol = outputCol + v;\n"
    "                        #if gPadZeros == 0\n"
    "                             inputCol += gHalfFilterSize;\n"
    "                        #endif\n"
    "                        sum += _inputPlane[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        // output are organized like [imageid][filterid][row][col]\n"
    "        int resultIndex = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId;\n"
    "        if (localId < gOutputSizeSquared) {\n"
    "            output[resultIndex ] = sum;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "forward_2_by_outplane", options, "cl/forward2.cl");
    // [[[end]]]
}

