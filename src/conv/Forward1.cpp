// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/Forward1.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Forward1::~Forward1() {
    delete kernel;
    delete addBias;
}
VIRTUAL void Forward1::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("Forward1::forward START");

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);

    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = std::min(globalSize, cl->getMaxWorkgroupSize());
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
//    cout << "forward1 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("Forward1::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward1::forward END");
}
Forward1::Forward1(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
    addBias = new AddBias(cl);

    std::string options = "";
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward1.cl", "convolve_imagecubes_float2", 'options')
    // ]]]
    // generated using cog, from cl/forward1.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// notes on non-odd filtersizes:\n"
    "// for odd, imagesize and filtersize 3, padZeros = 0:\n"
    "// output is a single square\n"
    "// m and n should vary between -1,0,1\n"
    "// for even, imagesize and filtersize 2, padzeros = 0\n"
    "// output is a single square, which we can position at topleft or bottomrigth\n"
    "// lets position it in bottomright\n"
    "// then m and n should vary as -1,0\n"
    "//\n"
    "// for even, imagesize and filtersize 2, padzeros = 1\n"
    "// output is 2 by 2\n"
    "// well... if it is even:\n"
    "// - if we are not padding zeros, then we simply move our filter around the image somehow\n"
    "// - if we are padding zeros, then we conceptually pad the bottom and right edge of the image with zeros by 1\n"
    "// filtersize remains the same\n"
    "//      m will vary as -1,0,1\n"
    "//       outputrow is fixed by globalid\n"
    "//       inputrow should be unchanged...\n"
    "// padzeros = 0:\n"
    "//  x x .  . . .\n"
    "//  x x .  . x x\n"
    "//  . . .  . x x\n"
    "// when filtersize even:\n"
    "//    new imagesize = oldimagesize - filtersize + 1\n"
    "// when filtersize odd:\n"
    "//    x x x .\n"
    "//    x x x .\n"
    "//    x x x .\n"
    "//    . . . .\n"
    "//    new imagesize = oldimagesize - filtersize + 1\n"
    "// padzeros = 1:\n"
    "// x x\n"
    "// x x . .   x x .    . . .     . . .\n"
    "//   . . .   x x .    . x x     . . .\n"
    "//   . . .   . . .    . x x     . . x x\n"
    "// outrow=0 outrow=1  outrow=2      x x\n"
    "// outcol=0 outcol=1  outcol=2    outrow=3\n"
    "//                                outcol=3\n"
    "// when filtersize is even, and padzeros, imagesize grows by 1 each time...\n"
    "//    imagesize = oldimagesize + 1\n"
    "// when filtersize is odd\n"
    "//  x x x\n"
    "//  x x x .   x x x    . . .\n"
    "//  x x x .   x x x    . x x x\n"
    "//    . . .   x x x    . x x x\n"
    "//                       x x x\n"
    "\n"
    "// images are organized like [imageId][plane][row][col]\n"
    "// filters are organized like [filterid][inplane][filterrow][filtercol]\n"
    "// output are organized like [imageid][filterid][row][col]\n"
    "// global id is organized like output, ie: [imageid][outplane][outrow][outcol]\n"
    "// - no local memory used currently\n"
    "// - each thread:\n"
    "//     - loads a whole upstream cube\n"
    "//     - loads a whole filter cube\n"
    "//     - writes one output...\n"
    "void kernel convolve_imagecubes_float2(\n"
    "    const int numExamples,\n"
    "      global const float *inputs, global const float *filters,\n"
    "    global float *output) {\n"
    "    int globalId = get_global_id(0);\n"
    "\n"
    "    int outputImage2Id = globalId / gOutputSizeSquared;\n"
    "    int exampleId = outputImage2Id / gNumFilters;\n"
    "    int filterId = outputImage2Id % gNumFilters;\n"
    "\n"
    "    // intraimage coords\n"
    "    int localid = globalId % gOutputSizeSquared;\n"
    "    int outputRow = localid / gOutputSize;\n"
    "    int outputCol = localid % gOutputSize;\n"
    "\n"
    "    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;\n"
    "    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;\n"
    "\n"
    "    float sum = 0;\n"
    "    if (exampleId < numExamples) {\n"
    "        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {\n"
    "            global float const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;\n"
    "            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;\n"
    "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    "                // trying to reduce register pressure...\n"
    "                #if gPadZeros == 1\n"
    "                    #define inputRowIdx (outputRow + u)\n"
    "                #else\n"
    "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    "                #endif\n"
    "                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    "                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    "                #pragma unroll\n"
    "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    "                    #if gPadZeros == 1\n"
    "                        #define inputColIdx (outputCol + v)\n"
    "                    #else\n"
    "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    "                    #endif\n"
    "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    "                    if (process) {\n"
    "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    if (exampleId < numExamples) {\n"
    "        output[globalId] = sum;\n"
    "    }\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "convolve_imagecubes_float2", options, "cl/forward1.cl");
    // [[[end]]]
}

