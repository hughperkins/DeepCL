// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/Forward4.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;
using namespace easycl;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Forward4::~Forward4() {
    delete kernel;
    delete addBias;
}
VIRTUAL void Forward4::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("Forward4::forward start");

    int numWorkgroups = dim.numFilters * batchSize * pixelsPerThread;
    int globalSize = workgroupSize * numWorkgroups;

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
    kernel->localFloats(square(dim.inputSize) );
    kernel->localFloats(square(dim.filterSize) );

    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();
    StatefulTimer::timeCheck("Forward4::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
}
Forward4::Forward4(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim)
            {
    addBias = new AddBias(cl);

    workgroupSize = std::max(32, square(dim.outputSize) ); // no point in wasting threads....
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    // see comments in forward4.cl,
    // if the outputimagesize * outputimagesize > maxWorkgroupSize,
    // then there wont be enough threads to process all the output points
    // of one image plane, so we create multiple workgroups for each
    // output image plane
    // here, we calculate how many workgroups we will need, in powers
    // of two:
    pixelsPerThread = 1;
    while(workgroupSize > maxWorkgroupSize) {
        workgroupSize = (workgroupSize + 1) >> 1;
        pixelsPerThread <<= 1;
    }
    //cout << "workgroupSize=" << workgroupSize << " pixelsPerThread=" << pixelsPerThread << endl;

    std::string options = ""; // "-D " + fn->getDefineName();
    options += " -D gWorkgroupSize=" + toString(workgroupSize);
    options += " -D gPixelsPerThread=" + toString(pixelsPerThread);
//    cout << "options " << options << endl;
    options += dim.buildOptionsString();
    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward4.cl", "forward_4_by_n_outplane_smallercache", 'options')
    // ]]]
    // generated using cog, from cl/forward4.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "void copyLocal(local float *target, global float const *source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "#ifdef gOutputSize // for previous tests that dont define it\n"
    "// workgroup id organized like: [n][filterid]\n"
    "// local id organized like: [outrow][outcol]\n"
    "// each thread iterates over: [upstreamplane][filterrow][filtercol]\n"
    "// number workgroups = 32\n"
    "// one filter plane takes up 5 * 5 * 4 = 100 bytes\n"
    "// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)\n"
    "// all filter cubes = 3.2KB * 32 = 102KB (too big)\n"
    "// output are organized like [n][filterid][outrow][outcol]\n"
    "// the pixels per thread thing... :\n"
    "// - we have one thread (~= cuda core) per output value,\n"
    "//   ie one thread for each combination of [outrow][outcol]\n"
    "// - however, the number of threads is typically limited on a gpu,\n"
    "//   eg to 512 (eg Intel HD), or 1024 (eg nVidia K520)\n"
    "// - so what happens if the number of output points is larger than\n"
    "//   the maximum workgroup size?\n"
    "// - then we have several possibilities really:\n"
    "//   - we can divide the image into blocks, and process each block\n"
    "//     separately.  This is probably a good option, but fair amount of\n"
    "//     work\n"
    "//   - we can get each thread to handle more than one output\n"
    "//     pixel, by looping\n"
    "//   - we can consider the output image in 1d, by putting the rows\n"
    "//     one after another, and assign each contiguous workgroup-size\n"
    "//     block to one workgroup\n"
    "//     => this is how this kernel works\n"
    "//     basically, it's a hack, so larger images actually run, without\n"
    "//     crashing, and we can probably improve it a lot :-)\n"
    "//\n"
    "// So, when outputSize * outputSize > workgroupSize, then\n"
    "// multiple workgroups will be created for each output plane\n"
    "// the number of such workgroups is given by: `gPixelsPerThread`\n"
    "// the id of our workgroup within such a set of workgroups is calculated\n"
    "// as `pixel`\n"
    "// effectiveLocalId is our local id if we had one enormous workgroup\n"
    "// containing the whole output image plane\n"
    "void kernel forward_4_by_n_outplane_smallercache(const int batchSize,\n"
    "      global const float *images, global const float *filters,\n"
    "    global float *output,\n"
    "    local float *_inputPlane, local float *_filterPlane) {\n"
    "    #define globalId (get_global_id(0))\n"
    "\n"
    "    #define localId (get_local_id(0))\n"
    "    #define workgroupId (get_group_id(0))\n"
    "//    const int workgroupSize = get_local_size(0);\n"
    "    const int effectiveWorkgroupId = workgroupId / gPixelsPerThread;\n"
    "    const int pixel = workgroupId % gPixelsPerThread;\n"
    "    const int effectiveLocalId = localId + pixel * gWorkgroupSize;\n"
    "    const int n = effectiveWorkgroupId / gNumFilters;\n"
    "    const int outPlane = effectiveWorkgroupId % gNumFilters;\n"
    "\n"
    "    const int outputRow = effectiveLocalId / gOutputSize;\n"
    "    const int outputCol = effectiveLocalId % gOutputSize;\n"
    "\n"
    "    float sum = 0;\n"
    "    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        copyLocal(_inputPlane, images + (n * gInputPlanes + upstreamPlane) * gInputSizeSquared, gInputSizeSquared);\n"
    "        copyLocal(_filterPlane, filters + (outPlane * gInputPlanes + upstreamPlane) * gFilterSizeSquared, gFilterSizeSquared);\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "        if (effectiveLocalId < gOutputSizeSquared) {\n"
    "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    "                // trying to reduce register pressure...\n"
    "                #if gPadZeros == 1\n"
    "                    #define inputRow (outputRow + u)\n"
    "                #else\n"
    "                    #define inputRow (outputRow + u + gHalfFilterSize)\n"
    "                #endif\n"
    "                int inputimagerowoffset = inputRow * gInputSize;\n"
    "                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    "                bool rowOk = inputRow >= 0 && inputRow < gInputSize;\n"
    "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    "                    #if gPadZeros == 1\n"
    "                        #define inputCol (outputCol + v)\n"
    "                    #else\n"
    "                        #define inputCol (outputCol + v + gHalfFilterSize)\n"
    "                    #endif\n"
    "                    bool process = rowOk && inputCol >= 0 && inputCol < gInputSize;\n"
    "                    if (process) {\n"
    "                            sum += _inputPlane[ inputimagerowoffset + inputCol] * _filterPlane[ filterrowoffset + v ];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    // output are organized like [imageid][filterid][row][col]\n"
    "    #define resultIndex (( n * gNumFilters + outPlane) * gOutputSizeSquared + effectiveLocalId)\n"
    "    if (effectiveLocalId < gOutputSizeSquared) {\n"
    "        output[resultIndex ] = sum;\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "forward_4_by_n_outplane_smallercache", options, "cl/forward4.cl");
    // [[[end]]]
}

