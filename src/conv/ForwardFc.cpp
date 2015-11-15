// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "conv/ForwardFc.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"
#include "conv/ReduceSegments.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardFc::~ForwardFc() {
    delete kernel1;
//    delete kernel_reduce;
    delete addBias;
    delete reduceSegments;
}
VIRTUAL void ForwardFc::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardFc::forward begin");

//    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();

    const int outputTotalSize = batchSize * dim.numFilters;
    const int output2Size = outputTotalSize * dim.numInputPlanes; // need to reduce over each input plane
    const int output1Size = output2Size * dim.filterSize; // need to reduce also over each row

//    const int output1Size = batchSize * dim.numFilters * dim.numInputPlanes * dim.filterSize;
    float *output1 = new float[ output1Size ];
    CLWrapper *output1Wrapper = cl->wrap(output1Size, output1);
    output1Wrapper->createOnDevice();

//    const int output2Size = batchSize * dim.numFilters * dim.numInputPlanes;
    float *output2 = new float[ output2Size ];
    CLWrapper *output2Wrapper = cl->wrap(output2Size, output2);
    output2Wrapper->createOnDevice();

    kernel1->in(batchSize);
    kernel1->input(dataWrapper);
    kernel1->input(weightsWrapper);
    kernel1->output(output1Wrapper);
    kernel1->localFloats(dim.inputSize);
    kernel1->localFloats(dim.numFilters * dim.filterSize  );

    int workgroupSize = dim.numFilters;
    // uncommenting next line causes out-of-bounds access currently:
    workgroupSize = (( workgroupSize + 32 - 1) / 32) * 32; // round up to nearest 32
    int numWorkgroups = dim.filterSize * dim.numInputPlanes;

    kernel1->run_1d(workgroupSize * numWorkgroups, workgroupSize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardFc::forward after first kernel");

    reduceSegments->reduce(output1Size, dim.filterSize, output1Wrapper, output2Wrapper);
    reduceSegments->reduce(output2Size, dim.numInputPlanes, output2Wrapper, outputWrapper);

    // add bias...
    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }

    delete output2Wrapper;
    delete[] output2;

    delete output1Wrapper;
    delete[] output1;
    StatefulTimer::timeCheck("ForwardFc::forward end");
}
ForwardFc::ForwardFc(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
            {

    if(dim.inputSize != dim.filterSize) {
        throw runtime_error("For ForwardFc, filtersize and inputimagesize must be identical");
    }
    if(dim.padZeros) {
        throw runtime_error("For ForwardFc, padzeros must be disabled");
    }

    this->addBias = new AddBias(cl);
    this->reduceSegments = new ReduceSegments(cl);

    std::string options = "";
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel1", "cl/forward_fc_wgperrow.cl", "forward_fc_workgroup_perrow", 'options')
    // # stringify.write_kernel2("kernel_reduce", "cl/reduce_segments.cl", "reduce_segments", 'options')
    // ]]]
    // generated using cog, from cl/forward_fc_wgperrow.cl:
    const char * kernel1Source =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "void copyLocal(local float *restrict target, global float const *restrict source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// concept:\n"
    "//  we want to share each input example across multiple filters\n"
    "//   but an entire filter plane is 19*19*4 = 1.4KB\n"
    "//   so eg 500 filter planes is 500* 1.4KB = 700KB, much larger than local storage\n"
    "//   of ~43KB\n"
    "//  - we could take eg 16 filters at a time, store one filter plane from each in local storage,\n"
    "//  and then bring down one example plane at a time, into local storage, during iteration over n\n"
    "//  - here though, we are going to store one row from one plane from each filter,\n"
    "//  and process against one row, from same plane, from each example\n"
    "//  so each workgroup will have one thread per filterId, eg 351 threads\n"
    "//    each thread will add up over its assigned row\n"
    "//  then, later we need to reduce over the rows\n"
    "//   ... and also over the input planes?\n"
    "//\n"
    "// workgroupid [inputplane][filterrow]\n"
    "// localid: [filterId]\n"
    "//  each thread iterates over: [n][filtercol]\n"
    "//  each thread is assigned to: one row, of one filter\n"
    "//  workgroup is assigned to: same row, from each input plane\n"
    "// local memory: one row from each output, = 128 * 19 * 4 = 9.8KB\n"
    "//             1 * input row = \"0.076KB\"\n"
    "// output1 structured as: [n][inputplane][filter][row], need to reduce again after\n"
    "// this kernel assumes:\n"
    "//   padzeros == 0 (mandatory)\n"
    "//   filtersize == inputimagesize (mandatory)\n"
    "//   inputimagesize == 19\n"
    "//   filtersize == 19\n"
    "//   outputSize == 1\n"
    "//   lots of outplanes/filters, hundreds, but less than max work groupsize, eg 350, 500, 361\n"
    "//   lots of inplanes, eg 32-128\n"
    "//   inputimagesize around 19, not too small\n"
    "#if (gFilterSize == gInputSize) && (gPadZeros == 0)\n"
    "void kernel forward_fc_workgroup_perrow(const int batchSize,\n"
    "    global const float *images, global const float *filters,\n"
    "    global float *output1,\n"
    "    local float *_imageRow, local float *_filterRows) {\n"
    "    const int globalId = get_global_id(0);\n"
    "\n"
    "    const int workgroupId = get_group_id(0);\n"
    "    const int workgroupSize = get_local_size(0);\n"
    "    const int localId = get_local_id(0);\n"
    "\n"
    "    const int inputPlaneId = workgroupId / gFilterSize;\n"
    "    const int filterRowId = workgroupId % gFilterSize;\n"
    "\n"
    "    const int filterId = localId;\n"
    "\n"
    "    // first copy down filter row, which is per-thread, so we have to copy it all ourselves...\n"
    "    global const float *filterRow = filters\n"
    "        + filterId * gNumInputPlanes * gFilterSizeSquared\n"
    "        + inputPlaneId * gFilterSizeSquared\n"
    "        + filterRowId * gFilterSize;\n"
    "    local float *_threadFilterRow = _filterRows + localId * gFilterSize;\n"
    "    if (localId < gNumFilters) {\n"
    "        for (int i = 0; i < gFilterSize; i++) {\n"
    "            _threadFilterRow[i] = filterRow[i];\n"
    "        }\n"
    "    }\n"
    "    const int loopsPerExample = (gInputSize + workgroupSize - 1) / workgroupSize;\n"
    "    // now loop over examples...\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        // copy down example row, which is global to all threads in workgroup\n"
    "        // hopefully should be enough threads....\n"
    "        // but we should check anyway really, since depends on number of filters configured,\n"
    "        // not on relative size of filter and input image\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        copyLocal(_imageRow,  images\n"
    "            + (( n\n"
    "                * gNumInputPlanes + inputPlaneId)\n"
    "                * gInputSize + filterRowId)\n"
    "                * gInputSize,\n"
    "            gInputSize);\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        // add up the values in our row...\n"
    "        // note: dont activate yet, since need to reduce again\n"
    "        // output structured as: [n][filter][inputplane][filterrow], need to reduce again after\n"
    "        if (localId < gNumFilters) {\n"
    "            float sum = 0;\n"
    "            for (int filterCol = 0; filterCol < gFilterSize; filterCol++) {\n"
    "                sum += _imageRow[ filterCol ] * _threadFilterRow[ filterCol ];\n"
    "            }\n"
    "            output1[ n * gNumInputPlanes * gNumFilters * gFilterSize\n"
    "                + inputPlaneId * gFilterSize\n"
    "                + filterId * gNumInputPlanes * gFilterSize + filterRowId ] = sum;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "";
    kernel1 = cl->buildKernelFromString(kernel1Source, "forward_fc_workgroup_perrow", options, "cl/forward_fc_wgperrow.cl");
    // [[[end]]]
}

