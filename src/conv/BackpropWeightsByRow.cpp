// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackpropWeightsByRow.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

#include "test/PrintBuffer.h"

using namespace std;
using namespace easycl;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsByRow::~BackpropWeightsByRow() {
    delete kernel;
    delete reduce;
    delete perElementAdd;
}
VIRTUAL void BackpropWeightsByRow::backpropWeights(int batchSize, float learningRate,  CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsByRow start");

    cout << "input buffer:" << endl;
    PrintBuffer::printFloats(cl, imagesWrapper, batchSize * dim.inputSize, dim.inputSize);
    cout << endl;

    cout << "errors buffer:" << endl;
    PrintBuffer::printFloats(cl, gradOutputWrapper, batchSize * dim.outputSize, dim.outputSize);
    cout << endl;

    int globalSize = workgroupSize * numWorkgroups;
    globalSize = (( globalSize + workgroupSize - 1) / workgroupSize) * workgroupSize;

    int localMemRequiredKB = (dim.outputSize * 4 + dim.inputSize * 4) / 1024 + 1;
    if(localMemRequiredKB >= cl->getLocalMemorySizeKB()) {
        throw runtime_error("local memory too small to use this kernel on this device.  Need: " + 
            toString(localMemRequiredKB) + "KB, but only have: " + 
            toString(cl->getLocalMemorySizeKB()) + "KB local memory");
    }

    const float learningMultiplier = learningRateToMultiplier(batchSize, learningRate);

    const int weights1Size = dim.filtersSize * dim.outputSize;
    float *weights1 = new float[ weights1Size ];
    CLWrapper *weights1Wrapper = cl->wrap(weights1Size, weights1);
    weights1Wrapper->createOnDevice();

    float *bias1 = 0;
    CLWrapper *bias1Wrapper = 0;
    if(dim.biased) {
        const int bias1Size = dim.numFilters * dim.outputSize;
        bias1 = new float[ bias1Size ];
        bias1Wrapper = cl->wrap(bias1Size, bias1);
        bias1Wrapper->createOnDevice();
    }

    float *weights2 = new float[ dim.filtersSize ];
    CLWrapper *weights2Wrapper = cl->wrap(dim.filtersSize, weights2);
    weights2Wrapper->createOnDevice();

    float *bias2 = 0;
    CLWrapper *bias2Wrapper = 0;
    if(dim.biased) {
        bias2 = new float[ dim.numFilters ];
        bias2Wrapper = cl->wrap(dim.numFilters, bias2);
        bias2Wrapper->createOnDevice();
    }

    StatefulTimer::instance()->timeCheck("BackpropWeightsByRow allocated buffers and wrappers");

    kernel
       ->in(learningMultiplier)
       ->in(batchSize)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper)
       ->out(weights1Wrapper);
    if(dim.biased) {
        kernel->out(bias1Wrapper);
    }
    kernel
        ->localFloats(dim.outputSize)
        ->localFloats(dim.inputSize);

    kernel->run_1d(globalSize, workgroupSize);
    cl->finish();

    cout << "weights1wrapper after first kernel:" << endl;
    PrintBuffer::printFloats(cl, weights1Wrapper, dim.filterSize * dim.outputSize, dim.filterSize);
    cout << endl;

    reduce->in(dim.filtersSize)->in(dim.outputSize)
        ->in(weights1Wrapper)->out(weights2Wrapper);
    reduce->run_1d(( dim.filtersSize + 64 - 1) / 64 * 64, 64);
    if(dim.biased) {
        reduce->in(dim.numFilters)->in(dim.outputSize)
            ->in(bias1Wrapper)->out(bias2Wrapper);
        reduce->run_1d(( dim.numFilters + 64 - 1) / 64 * 64, 64);
    }
    cl->finish();

    PrintBuffer::printFloats(cl, weights2Wrapper, dim.filterSize, dim.filterSize);

    PrintBuffer::printFloats(cl, weightsWrapper, dim.filterSize, dim.filterSize);
    
    perElementAdd->in(dim.filtersSize)->inout(weightsWrapper)->in(weights2Wrapper);
    perElementAdd->run_1d(( dim.filtersSize + 64 - 1) / 64 * 64, 64);
    
    if(dim.biased) {
        perElementAdd->in(dim.numFilters)->inout(biasWrapper)->in(bias2Wrapper);
        perElementAdd->run_1d(( dim.numFilters + 64 - 1) / 64 * 64, 64);
    }

    cl->finish();

    PrintBuffer::printFloats(cl, weightsWrapper, dim.filterSize, dim.filterSize);

    if(dim.biased) {
        delete bias2Wrapper;
        delete bias1Wrapper;
        delete[] bias2;
        delete[] bias1;
    }
    delete weights2Wrapper;
    delete weights1Wrapper;
    delete[] weights2;
    delete[] weights1;

    StatefulTimer::instance()->timeCheck("BackpropWeightsByRow end");
}
BackpropWeightsByRow::BackpropWeightsByRow(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    workgroupSize = std::max(32, dim.filterSize); // no point in wasting cores...
    numWorkgroups = dim.inputPlanes * dim.numFilters * dim.outputSize;
    cout << "numWorkgroups " << numWorkgroups << " workgropuSize=" << workgroupSize << endl;
    if(workgroupSize > cl->getMaxWorkgroupSize()) {
        throw runtime_error("filtersize larger than maxworkgroupsize, so cannot use BackpropWeightsByRow kernel");
    }    

    std::string options = dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/backpropweights_byrow.cl", "backprop_weights", 'options')
    // stringify.write_kernel2("reduce", "cl/reduce_segments.cl", "reduce_segments", '""')
    // stringify.write_kernel2("perElementAdd", "cl/per_element_add.cl", "per_element_add", '""')
    // ]]]
    // generated using cog, from cl/backpropweights_byrow.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// reminder:\n"
    "// - for backprop weights, we take one plane from one image, convolve with one plane from the output\n"
    "//   and reduce over n\n"
    "\n"
    "// concept:\n"
    "// - here, we process only single row from the input/output cube (same row from each)\n"
    "//   and then we will need to reduce the resulting weight changes over the rows, in a separate kernel\n"
    "// - this assumes that the filter cubes are small, so reducing over 32 or so of them is not a big task\n"
    "\n"
    "// this isnt expected to give good performance, but it paves the way for creating workgroups with\n"
    "// multiple pairs of input/output planes in, which might reduce memory copying from global\n"
    "// filters themselves are fairly small, and plasuibly easy to reduce?\n"
    "\n"
    "// here, we will use one workgroup for one row of a single pair of input/output planes\n"
    "// and sum over n\n"
    "// workgroup: [outputPlane][inputPlane][outputRow]\n"
    "// localid: [filterRow][filterCol]\n"
    "// weightChanges1: [outputPlane][inputPlane][filterRow][filterCol][outputRow]\n"
    "// gradBiasWeights1: [outputPlane][outputRow]\n"
    "kernel void backprop_weights(const float learningRateMultiplier, const int batchSize,\n"
    "    global float const *gradOutput, global float const *input, global float *restrict gradWeights1,\n"
    "    #ifdef BIASED\n"
    "         global float *restrict gradBiasWeights1,\n"
    "    #endif\n"
    "    local float *restrict _errorRow, local float *restrict _inputRow) {\n"
    "    #define globalId (get_global_id(0))\n"
    "    #define workgroupId (get_group_id(0))\n"
    "    #define localId (get_local_id(0))\n"
    "\n"
    "    const int filterRow = localId / gFilterSize;\n"
    "    const int filterCol = localId % gFilterSize;\n"
    "    const int outputRow = workgroupId % gOutputSize;\n"
    "    #define outInCombo (workgroupId / gOutputSize)\n"
    "    const int outputPlane = outInCombo / gNumInputPlanes;\n"
    "    const int inputPlane = outInCombo % gNumInputPlanes;\n"
    "\n"
    "    const int thisInputRow = outputRow - gMargin; // + filterRow;\n"
    "\n"
    "    float thiswchange = 0.0f;\n"
    "    #ifdef BIASED\n"
    "        float thisbiaschange = 0.0f;\n"
    "    #endif\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        // copy down the gradOutput row...\n"
    "        {\n"
    "            global float const*gradOutputRow = gradOutput +\n"
    "                (( n\n"
    "                    * gNumOutputPlanes + outputPlane)\n"
    "                    * gOutputSize + outputRow)\n"
    "                    * gOutputSize;\n"
    "            if (localId < gOutputSize) { // assume we have enough threads for now... should fix later\n"
    "                _errorRow[ localId ] = gradOutputRow[ localId ];\n"
    "            }\n"
    "        }\n"
    "        // copy down the input row\n"
    "        {\n"
    "            global float const*inputRowData = input +\n"
    "                (( n\n"
    "                    * gNumInputPlanes + inputPlane)\n"
    "                    * gInputSize + thisInputRow)\n"
    "                    * gInputSize;\n"
    "            if (localId < gInputSize) { // assume we have enough threads for now... should fix later\n"
    "                _inputRow[ localId ] = inputRowData[ localId ];\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int outputCol = 0; outputCol < gOutputSize; outputCol++) {\n"
    "            const int inputCol = outputCol - gMargin + filterCol;\n"
    "            if (inputRow >= 0 && inputRow < gInputSize && inputCol >= 0 && inputCol < gInputSize) {\n"
    "                if (localId < gFilterSizeSquared) {\n"
    "                    thiswchange += _inputRow[ inputCol ] * _errorRow[ outputCol ];\n"
    "                    #ifdef BIASED\n"
    "                        thisbiaschange += _errorRow[ outputCol ];\n"
    "                    #endif\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    if (workgroupId == 0 && localId == 0) {\n"
    "        gradWeights1[0] = _inputRow[0];\n"
    "        gradWeights1[1] = _inputRow[1];\n"
    "    }\n"
    "\n"
    "    if (localId < gFilterSizeSquared) {\n"
    "        #define weightsIndex (( (outInCombo \\\n"
    "            * gFilterSizeSquared) + localId \\\n"
    "            * gOutputSize) + outputRow)\n"
    "        //gradWeights1[ weightsIndex ] -= learningRateMultiplier * thiswchange;\n"
    "        //gradWeights1[weightsIndex] = 123.0f;\n"
    "    }\n"
    "    #ifdef BIASED\n"
    "        if (inputPlane == 0 && localId == 0) {\n"
    "            gradBiasWeights1[outputPlane * gOutputSize + outputRow ] = learningRateMultiplier * thisbiaschange;\n"
    "        }\n"
    "    #endif\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backprop_weights", options, "cl/backpropweights_byrow.cl");
    // generated using cog, from cl/reduce_segments.cl:
    const char * reduceSource =  
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
    reduce = cl->buildKernelFromString(reduceSource, "reduce_segments", "", "cl/reduce_segments.cl");
    // generated using cog, from cl/per_element_add.cl:
    const char * perElementAddSource =  
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
    perElementAdd = cl->buildKernelFromString(perElementAddSource, "per_element_add", "", "cl/per_element_add.cl");
    // [[[end]]]
}

