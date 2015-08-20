// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "ForwardFc_workgroupPerFilterPlane.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardFc_workgroupPerFilterPlane::~ForwardFc_workgroupPerFilterPlane() {
    delete kernel1;
    delete kernel2;
}
VIRTUAL void ForwardFc_workgroupPerFilterPlane::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardFc_workgroupPerFilterPlane::forward begin");
    const int output1Size = batchSize * dim.numFilters * dim.filterSize;
    float *output1 = new float[ output1Size ];
    CLWrapper *output1Wrapper = cl->wrap(output1Size, output1);
    output1Wrapper->createOnDevice();

    kernel1->in(batchSize);
    kernel1->input(dataWrapper);
    kernel1->input(weightsWrapper);
    if(dim.biased) kernel1->input(biasWrapper);
    kernel1->output(output1Wrapper);
    kernel1->localFloats(dim.inputSize);
    kernel1->localFloats(batchSize * dim.filterSize);

    int workgroupSize = dim.numFilters;
    int numWorkgroups = dim.filterSize;

    int globalSize = workgroupSize * numWorkgroups;
/////    cout << "forward3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel1->run_1d(globalSize, workgroupSize);
    cl->finish();
    StatefulTimer::timeCheck("ForwardFc_workgroupPerFilterPlane::forward after first kernel");

    // now reduce again...
    kernel2->in(batchSize)->in(output1Wrapper)->out(outputWrapper);
    int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    numWorkgroups = (batchSize * dim.numFilters + maxWorkgroupSize - 1) / maxWorkgroupSize;
    kernel2->run_1d(numWorkgroups * maxWorkgroupSize, maxWorkgroupSize);
    cl->finish();

    delete output1Wrapper;
    delete[] output1;
    StatefulTimer::timeCheck("ForwardFc_workgroupPerFilterPlane::forward end");
}
ForwardFc_workgroupPerFilterPlane::ForwardFc_workgroupPerFilterPlane(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
            {

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel1", "cl/forward_fc_wgperrow.cl", "forward_fc_workgroup_perrow", 'options')
    // stringify.write_kernel2("kernel2", "cl/forward_fc.cl", "reduce_rows", 'options')
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
    "    for(int loop = 0; loop < numLoops; loop++) {\n" 
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n" 
    "        if(offset < N) {\n" 
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
    "    for(int i = 0; i < gFilterSize; i++) {\n" 
    "        _threadFilterRow[i] = filterRow[i];\n" 
    "    }\n" 
    "    const int loopsPerExample = (gInputSize + workgroupSize - 1) / workgroupSize;\n" 
    "    // now loop over examples...\n" 
    "    for(int n = 0; n < batchSize; n++) {\n" 
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
    "        float sum = 0;\n" 
    "        for(int filterCol = 0; filterCol < gFilterSize; filterCol++) {\n" 
    "            sum += _imageRow[ filterCol ] * _threadFilterRow[ filterCol ];\n" 
    "        }\n" 
    "        // note: dont activate yet, since need to reduce again\n" 
    "        // output structured as: [n][filter][inputplane][filterrow], need to reduce again after\n" 
    "        if(localId < gNumFilters) {\n" 
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
    // generated using cog, from cl/forward_fc.cl:
    const char * kernel2Source =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "#ifdef gOutImageSize // for previous tests that dont define it\n" 
    "// workgroupid [n][outputplane]\n" 
    "// localid: [filterrow][filtercol]\n" 
    "//  each thread iterates over: [inplane]\n" 
    "// this kernel assumes:\n" 
    "//   padzeros == 0 (mandatory)\n" 
    "//   filtersize == inputimagesize (mandatory)\n" 
    "//   outputSize == 1\n" 
    "//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361\n" 
    "//   lots of inplanes, eg 32\n" 
    "//   inputimagesize around 19, not too small\n" 
    "#if gFilterSize == gInputSize && gPadZeros == 0\n" 
    "void kernel forward_filter_matches_inimage(const int batchSize,\n" 
    "      global const float *images, global const float *filters,\n" 
    "    global float *output,\n" 
    "    local float *_upstreamImage, local float *_filterImage) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int n = workgroupId / gNumOutPlanes;\n" 
    "    const int outPlane = workgroupId % gNumOutPlanes;\n" 
    "\n" 
    "    const int localId = get_local_id(0);\n" 
    "    const int filterRow = localId / gFilterSize;\n" 
    "    const int filterCol = localId % gFilterSize;\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    for(int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++) {\n" 
    "        int thisUpstreamImageOffset = (n * gUpstreamNumPlanes + upstreamPlane) * gUpstreamImageSizeSquared;\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        for(int i = 0; i < numUpstreamsPerThread; i++) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if(thisOffset < gUpstreamImageSizeSquared) {\n" 
    "                _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];\n" 
    "            }\n" 
    "        }\n" 
    "        const int filterGlobalOffset = (outPlane * gUpstreamNumPlanes + upstreamPlane) * gFilterSizeSquared;\n" 
    "        for(int i = 0; i < numFilterPixelsPerThread; i++) {\n" 
    "            int thisOffset = workgroupSize * i + localId;\n" 
    "            if(thisOffset < gFilterSizeSquared) {\n" 
    "                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];\n" 
    "            }\n" 
    "        }\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        if(localId < gOutImageSizeSquared) {\n" 
    "            for(int u = minu; u <= maxu; u++) {\n" 
    "                int inputRow = outputRow + u + (gPadZeros ? 0 : gHalfFilterSize);\n" 
    "                int inputimagerowoffset = inputRow * gUpstreamImageSize;\n" 
    "                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                for(int v = minv; v <= maxv; v++) {\n" 
    "                    int inputCol = outputCol + v + (gPadZeros ? 0 : gHalfFilterSize);\n" 
    "                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "    // output are organized like [imageid][filterid][row][col]\n" 
    "    int resultIndex = (n * gNumOutPlanes + outPlane) * gOutImageSizeSquared + localId;\n" 
    "    if(localId < gOutImageSizeSquared) {\n" 
    "        output[resultIndex ] = sum;\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "#endif\n" 
    "\n" 
    "\n" 
    "";
    kernel2 = cl->buildKernelFromString(kernel2Source, "reduce_rows", options, "cl/forward_fc.cl");
    // [[[end]]]
}

