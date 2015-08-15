// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ForwardIm2Col.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
#define PUBLIC

static void im2col(CLWrapper* im, int imOffset, const int channels,
    const int size, const int ksize, const int padding, const int stride, CLWrapper* columns) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int size_col = (size + 2 * padding - ksize) / stride + 1;
  int num_kernels = channels * size_col * size_col;

  std::string uniqueName = "SpatialConvolutionMM::im2col";
  EasyCL *cl = im->storage->cl;
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialConvolutionMM.cl",
      SpatialConvolutionMM_getKernelTemplate(), "im2col_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(im);
  k.in(height);
  k.in(width);
  k.in(ksize_h);
  k.in(ksize_w);
  k.in(pad_h);
  k.in(pad_w);
  k.in(stride_h);
  k.in(stride_w);
  k.in(height_col);
  k.in(width_col);
  k.out(col);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}

void col2im(THClState *state, THClTensor* col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, THClTensor* im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.

  EasyCL *cl = im->storage->cl;
  std::string uniqueName = "SpatialConvolutionMM::col2im";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernel = kernelBuilder.buildKernel(uniqueName, "SpatialConvolutionMM.cl",
      SpatialConvolutionMM_getKernelTemplate(), "col2im_kernel");
  }

  THClKernels k(state, kernel);
  k.in(num_kernels);
  k.in(col);
  k.in(height);
  k.in(width);
  k.in(channels);

  k.in(patch_h);
  k.in(patch_w);
  k.in(pad_h);
  k.in(pad_w);
  k.in(stride_h);
  k.in(stride_w);

  k.in(height_col);
  k.in(width_col);
  k.out(im);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}

PUBLIC VIRTUAL ForwardIm2Col::~ForwardIm2Col() {
    delete kernel;
    delete addBias;
	delete columnsWrapper;
	delete onesWrapper;
	delete columns;
	delete ones;
}
PUBLIC VIRTUAL void ForwardIm2Col::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("ForwardIm2Col::forward START");

// =====================

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b ++) {
    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m_ = dim.numFilters;
    long n_ = dim.outputImageSizeSquared;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cl_err err = clblasSgemm(clblasColumnMajor, clblasTrans, clblasNoTrans, n_, m_, k_,
                         1, onesWrapper->getBuffer(), 0, k_,
                         biasWrapper->getBuffer(), 0, k_,
						 0,
                         outputWrapper->getBuffer(), b * dim.outputCubeSize, n_,
                         1, cl->queue, 0, NULL, 0);
    if (err != CL_SUCCESS) {
        throw runtime_error("clblasSgemm() failed with " + toString(err));
    }

    // Extract columns:
    im2col(
      state,
      input_n,
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      columns
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THClBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        columns, n,
        weight, k,
        1,
        output_n, n
    );
  }

  // Free
  THClTensor_free(state, input_n);
  THClTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THClTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
    THClTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }


//=========================

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->output( outputWrapper );

    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = std::min( globalSize, cl->getMaxWorkgroupSize() );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//    cout << "forward1 globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
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

	int columnsSize= dim.inputPlanes * dim.filterSizeSquared * dim.outputImageSizeSquared;
	columns = new float[];
	columns = cl->wrap(columnsSize, columns);

	int onesSize = dim.outputImageSizeSquared;
	ones = new float[onesSize];
	onesWrapper = cl->wrap(onesSize, ones);

    std::string options = "";
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/ForwardIm2Col.cl", "convolve_imagecubes_float2", 'options' )
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
}

