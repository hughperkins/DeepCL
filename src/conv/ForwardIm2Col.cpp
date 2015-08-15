// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ForwardIm2Col.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

#include <sstream>
#include <iostream>
#include <string>

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
#define PUBLIC

STATIC void ForwardIm2Col::im2col(
        CLWrapper* im, int imOffset, CLWrapper* columns) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int num_kernels = channels * size_col * size_col;

    CLKernel *k = kernelIm2Col;
    k->in(num_kernels);
    k->in(im);
    k->out(col);

    k->run_1d(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}
//STATIC void ForwardIm2Col::col2im(
//        CLWrapper* col, THClTensor* im) {
//    int num_kernels = channels * height * width;
//    // To avoid involving atomic operations, we will launch one kernel per
//    // bottom dimension, and then in the kernel add up the top dimensions.

//    EasyCL *cl = im->storage->cl;
//    std::string uniqueName = "ForwardIm2Col::col2im";
//    CLKernel *kernel = 0;
//    if(cl->kernelExists(uniqueName)) {
//    kernel = cl->getKernel(uniqueName);
//    } else {
//    TemplatedKernel kernelBuilder(cl);
//    kernel = kernelBuilder.buildKernel(uniqueName, "ForwardIm2Col.cl",
//      ForwardIm2Col_getKernelTemplate(), "col2im_kernel");
//    }

//    CLKernel *k = kernelCol2Im;
//    k->in(num_kernels);
//    k->in(col);
//    k->out(im);

//    k->run_1d(GET_BLOCKS(state, num_kernels), getNumThreads(state));
//}
PUBLIC VIRTUAL ForwardIm2Col::~ForwardIm2Col() {
    delete kernelIm2Col;
//    delete kernelCol2Im;
    delete addBias;
}
PUBLIC VIRTUAL void ForwardIm2Col::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("ForwardIm2Col::forward START");

    int columnsSize= dim.inputPlanes * dim.filterSizeSquared * dim.outputImageSizeSquared;
    float *columns = new float[columnsSize];
    CLWrapper *columnsWrapper = cl->wrap(columnsSize, columns);

    StatefulTimer::timeCheck("ForwardIm2Col::forward after alloc");

    for (int b = 0; b < batchSize; b ++) {
        // Extract columns:
        im2col(
            dataWrapper,
            b * inputCubeSize,
            columnsWrapper
        );

        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
        long m = dim.numFilters;
        long n = dim.outputSizeSquared;
        long k = dim.numFilters * dim.filterSizeSquared;

        // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
        cl_err err = clblasSgemm(
            clblasColumnMajor,
            clblasNoTrans, clblasNoTrans,
            n, m, k,
            1,
            columnsWrapper->getBuffer(), 0, n,
            weightsWrapper->getBuffer(), 0, k,
            1,
            outputWrapper->getBuffer(), b * dim.outputCubeSize, n,
            1, cl->queue, 0, NULL, 0
        );
        if (err != CL_SUCCESS) {
            throw runtime_error("clblasSgemm() failed with " + toString(err));
        }
    }

    delete columnsWrapper;
    delete columns;

    StatefulTimer::timeCheck("ForwardIm2Col::forward after call forward");

    if( dim.biased ) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputImageSize,
            outputWrapper, biasWrapper );
    }
    StatefulTimer::timeCheck("ForwardIm2Col::forward END");
}
PUBLIC ForwardIm2Col::ForwardIm2Col( EasyCL *cl, LayerDimensions dim ) :
            Forward( cl, dim )
        {
    addBias = new AddBias( cl );

    int size = dim.inputSize;
    int padding = dim.padZeros ? dim.halfFilterSize : 0;
    int stride = 1;
    int size_col = (size + 2 * padding - filterSize) / stride + 1;

    TemplatedKernel builder(cl);
    builder.set("padding", dim.padZeros ? dim.halfFilterSize : 0);
    builder.set("stride", 1);
    builder.set("colSize", size_col);
    builder.set("channels", dim.inputPlanes);
    builder.set("filterSize", dim.filterSize);
    builder.set("size", dim.inputImageSize);
    this->kernelIm2Col = kernelBuilder.buildKernel(
        "im2col",
        "ForwardIm2Col.cl",
        getIm2ColTemplate(),
        "im2col",
        false);
//    this->kernelCol2Im = kernelBuilder.buildKernel(
//        "col2im",
//        "ForwardIm2Col.cl",
//        getIm2ColTemplate(),
//        "col2im",
//        false);
}
STATIC std::string ForwardIm2Col::getKernelTemplate() {
    // [[[cog
    // import stringify
    // stringify.write_kernel( "kernel", "ForwardIm2Col.cl" )
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
    "    global float *output ) {\n" 
    "    int globalId = get_global_id(0);\n" 
    "\n" 
    "    int outputImage2Id = globalId / gOutputImageSizeSquared;\n" 
    "    int exampleId = outputImage2Id / gNumFilters;\n" 
    "    int filterId = outputImage2Id % gNumFilters;\n" 
    "\n" 
    "    // intraimage coords\n" 
    "    int localid = globalId % gOutputImageSizeSquared;\n" 
    "    int outputRow = localid / gOutputImageSize;\n" 
    "    int outputCol = localid % gOutputImageSize;\n" 
    "\n" 
    "    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputImageSizeSquared;\n" 
    "    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    if( exampleId < numExamples ) {\n" 
    "        for( int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++ ) {\n" 
    "            global float const*inputPlane = inputCube + inputPlaneIdx * gInputImageSizeSquared;\n" 
    "            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;\n" 
    "            for( int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++ ) {\n" 
    "                // trying to reduce register pressure...\n" 
    "                #if gPadZeros == 1\n" 
    "                    #define inputRowIdx ( outputRow + u )\n" 
    "                #else\n" 
    "                    #define inputRowIdx ( outputRow + u + gHalfFilterSize )\n" 
    "                #endif\n" 
    "                global float const *inputRow = inputPlane + inputRowIdx * gInputImageSize;\n" 
    "                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputImageSize;\n" 
    "                #pragma unroll\n" 
    "                for( int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++ ) {\n" 
    "                    #if gPadZeros == 1\n" 
    "                        #define inputColIdx ( outputCol + v )\n" 
    "                    #else\n" 
    "                        #define inputColIdx ( outputCol + v + gHalfFilterSize )\n" 
    "                    #endif\n" 
    "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputImageSize;\n" 
    "                    if( process ) {\n" 
    "                            sum += inputRow[inputColIdx] * filterRow[v];\n" 
    "                    }\n" 
    "                }\n" 
    "            }\n" 
    "        }\n" 
    "    }\n" 
    "\n" 
    "    if( exampleId < numExamples ) {\n" 
    "        output[globalId] = sum;\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "convolve_imagecubes_float2", options, "cl/ForwardIm2Col.cl" );
    // [[[end]]]
    return kernelSource;
}

