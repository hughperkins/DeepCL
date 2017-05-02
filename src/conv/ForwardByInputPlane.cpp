// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "ForwardByInputPlane.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

using namespace std;
using namespace easycl;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardByInputPlane::~ForwardByInputPlane() {
    delete kernel;
    delete reduceSegments;
    delete repeatedAdd;
//    delete activate;
}
VIRTUAL void ForwardByInputPlane::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardByInputPlane::forward begin");
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int maxglobalId = 0;

    int MBAllocRequired = (int) ((long)batchSize * dim.numFilters * dim.outputSizeSquared * dim.numInputPlanes * 4 / 1024 / 1024);
    if(MBAllocRequired >= cl->getMaxAllocSizeMB()) {
        throw runtime_error("memallocsize too small to use this kernel on this device.  Need: " + 
            toString(MBAllocRequired) + "MB, but only have: " + 
            toString(cl->getMaxAllocSizeMB()) + "MB max alloc size");
    }

    // [n][filterId][outRow][outCol][inputPlane]
    int output1Size = batchSize * dim.numFilters * dim.outputSizeSquared * dim.numInputPlanes;
//    cout << "output1size: " << output1Size << endl;
    float *output1 = new float[output1Size];
    CLWrapper *output1Wrapper = cl->wrap(output1Size, output1);

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(output1Wrapper);
    kernel->localFloats(square(dim.inputSize) );
    kernel->localFloats(square(dim.filterSize) * dim.numFilters);

    int workgroupsize = std::max(32, dim.numFilters * dim.outputSize); // no point in wasting threads....
    while(workgroupsize > cl->getMaxWorkgroupSize()) {
        workgroupsize >>= 1;
    }
    int numWorkgroups = dim.numInputPlanes;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "forwardbyinputplane numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << " numinputplanes=" << dim.numInputPlanes << endl;
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardByInputPlane::forward after kernel1");

//    {
//        output1Wrapper->copyToHost();
//        for(int i = 0; i < output1Size + 10; i++) {
//            cout << "output1[" << i << "]=" << output1[i] << " " << (i < output1Size) << endl;
//        }
//    }

    reduceSegments->in(batchSize * dim.numFilters * dim.outputSizeSquared)->in(dim.numInputPlanes)->in(output1Wrapper)->out(outputWrapper);
    maxglobalId = batchSize * dim.numFilters * dim.outputSize * dim.outputSize;
    numWorkgroups = (maxglobalId + maxWorkgroupSize - 1) / maxWorkgroupSize;
    reduceSegments->run_1d(numWorkgroups * maxWorkgroupSize, maxWorkgroupSize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardByInputPlane::forward after reduce over inputplanes");

    if(dim.biased) {
        repeatedAdd->in(batchSize * dim.numFilters * dim.outputSize * dim.outputSize)
            ->in(dim.numFilters)
            ->in(dim.outputSize * dim.outputSize)
            ->inout(outputWrapper)->in(biasWrapper);
        maxglobalId = batchSize * dim.numFilters * dim.outputSize * dim.outputSize;
        numWorkgroups = (maxglobalId + maxWorkgroupSize - 1) / maxWorkgroupSize;
        repeatedAdd->run_1d(numWorkgroups * maxWorkgroupSize, maxWorkgroupSize);
        cl->finish();
        StatefulTimer::timeCheck("ForwardByInputPlane::forward after repeatedAdd");
    }

//    activate->in(batchSize * dim.numFilters * dim.outputSize * dim.outputSize)
//        ->inout(outputWrapper);
//    maxglobalId = batchSize * dim.numFilters * dim.outputSize * dim.outputSize;
//    numWorkgroups = (maxglobalId + maxWorkgroupSize - 1) / maxWorkgroupSize;
//    activate->run_1d(numWorkgroups * maxWorkgroupSize, maxWorkgroupSize);
//    cl->finish();
//    StatefulTimer::timeCheck("ForwardByInputPlane::forward after activate");

    delete output1Wrapper;
    delete[] output1;

    StatefulTimer::timeCheck("ForwardByInputPlane::forward after call forward");
}
ForwardByInputPlane::ForwardByInputPlane(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
            {

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward_byinputplane.cl", "forward_byinputplane", 'options')
    // stringify.write_kernel2("reduceSegments", "cl/reduce_segments.cl", "reduce_segments", 'options')
    // stringify.write_kernel2("repeatedAdd", "cl/per_element_add.cl", "repeated_add", 'options')
    // # stringify.write_kernel2("activate", "cl/activate.cl", "activate", 'options')
    // ]]]
    // generated using cog, from cl/forward_byinputplane.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// concept:\n"
    "// - load same input plane from each image\n"
    "// - hold filter plane for this input plane, for all filters\n"
    "// - reduce afterwards\n"
    "// local memory for one plane from each filter of 64c7 = 64 * 7 * 7 * 4 = 12.5KB\n"
    "// local memory for one single input plane = 19 * 19 * 4 = 1.4KB\n"
    "// => seems ok?\n"
    "// workgroupid: [inputPlaneId]\n"
    "// localid: [filterId][outRow] (if this is more than workgroupsize, we should reuse some threads...)\n"
    "// iterate over: [n][outCol]\n"
    "// output: [n][filterId][outRow][outCol][inputPlane]\n"
    "// need to later reduce output over: [inputPlane]\n"
    "void kernel forward_byinputplane(const int batchSize,\n"
    "      global const float *images, global const float *filters,\n"
    "    global float *output,\n"
    "    local float *_inputPlane, local float *_filterPlanes) {\n"
    "//    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;\n"
    "\n"
    "    const int globalId = get_global_id(0);\n"
    "    const int workgroupId = get_group_id(0);\n"
    "    const int workgroupSize = get_local_size(0);\n"
    "    const int localId = get_local_id(0);\n"
    "\n"
    "    const int inputPlaneId = workgroupId;\n"
    "    const int numLoops = (gNumFilters * gOutputSize + workgroupSize - 1) / workgroupSize;\n"
    "    const int numFilterCopyLoops = (gFilterSizeSquared + gOutputSize - 1) / gOutputSize;\n"
    "    const int numImageCopyLoops = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        const int loopLocalId = localId + loop * workgroupSize;\n"
    "        const int filterId = loopLocalId / gOutputSize;\n"
    "        const int outRow = loopLocalId % gOutputSize;\n"
    "\n"
    "        // copy down our filter, we have gOutputSize threads to do this\n"
    "        global float const *globalFilterPlane = filters +\n"
    "            (filterId * gNumInputPlanes + inputPlaneId) * gFilterSizeSquared;\n"
    "        local float *_localFilterPlane = _filterPlanes + filterId * gFilterSizeSquared;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int i = 0; i < numFilterCopyLoops; i++) {\n"
    "            const int offset = i * gOutputSize + outRow;\n"
    "            bool process = filterId < gNumFilters && offset < gFilterSizeSquared;\n"
    "            if (process) {\n"
    "                _localFilterPlane[ offset ] = globalFilterPlane[ offset ];\n"
    "            }\n"
    "        }\n"
    "        // loop over n ...\n"
    "        for (int n = 0; n < batchSize; n++) {\n"
    "            // copy down our imageplane, we have workgroupSize threads to do this\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            global float const *globalImagePlane = images +\n"
    "                (n * gNumInputPlanes + inputPlaneId) * gInputSizeSquared;\n"
    "            for (int i = 0; i< numImageCopyLoops; i++) {\n"
    "                const int offset = i * workgroupSize + localId;\n"
    "                if (offset < gInputSizeSquared) {\n"
    "                    _inputPlane[ offset ] = globalImagePlane[ offset ];\n"
    "                }\n"
    "            }\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            // calc output for each [outrow][outcol]\n"
    "            bool filterPlaneOk = filterId < gNumFilters;\n"
    "            for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
    "                float sum = 0;\n"
    "                for (int filterRow = 0; filterRow < gFilterSize; filterRow++) {\n"
    "                    int inRow = outRow + filterRow;\n"
    "                    #if gPadZeros == 1\n"
    "                        inRow -= gHalfFilterSize;\n"
    "                    #endif\n"
    "                    bool rowOk = filterPlaneOk && inRow >= 0 && inRow < gInputSize;\n"
    "                    for (int filterCol = 0; filterCol < gFilterSize; filterCol++) {\n"
    "                        int inCol = outCol + filterCol;\n"
    "                        #if gPadZeros == 1\n"
    "                            inCol -= gHalfFilterSize;\n"
    "                        #endif\n"
    "                        bool process = rowOk && inCol >= 0 && inCol < gInputSize;\n"
    "                        if (process) {\n"
    "                            float imageValue = _inputPlane[ inRow * gInputSize + inCol ];\n"
    "                            float filterValue = _localFilterPlane[ filterRow * gFilterSize + filterCol ];\n"
    "                            sum += imageValue * filterValue;\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "                if (filterId < gNumFilters) {\n"
    "                    // [n][filterId][outRow][outCol][inputPlane]\n"
    "                    int resultIndex = (( (n\n"
    "                        * gNumFilters + filterId)\n"
    "                        * gOutputSize + outRow)\n"
    "                        * gOutputSize + outCol)\n"
    "                        * gNumInputPlanes + inputPlaneId;\n"
    "                    output[resultIndex] = sum;\n"
    "                    //if (globalId == 2) output[0] = resultIndex;\n"
    "//                    output[resultIndex] = outRow;\n"
    "                }\n"
    "//                output[localId] = _localFilterPlane[localId];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "forward_byinputplane", options, "cl/forward_byinputplane.cl");
    // generated using cog, from cl/reduce_segments.cl:
    const char * reduceSegmentsSource =  
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
    reduceSegments = cl->buildKernelFromString(reduceSegmentsSource, "reduce_segments", options, "cl/reduce_segments.cl");
    // generated using cog, from cl/per_element_add.cl:
    const char * repeatedAddSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "kernel void per_element_add(const int N, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[globalId];\n"
    "}\n"
    "\n"
    "// adds source to target\n"
    "// tiles source as necessary, according to tilingSize\n"
    "kernel void per_element_tiled_add(const int N, const int tilingSize, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[globalId % tilingSize];\n"
    "}\n"
    "\n"
    "kernel void repeated_add(const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source) {\n"
    "    const int globalId = get_global_id(0);\n"
    "    if (globalId >= N) {\n"
    "        return;\n"
    "    }\n"
    "    target[globalId] += source[ (globalId / repeatSize) % sourceSize ];\n"
    "}\n"
    "\n"
    "";
    repeatedAdd = cl->buildKernelFromString(repeatedAddSource, "repeated_add", options, "cl/per_element_add.cl");
    // [[[end]]]
}

