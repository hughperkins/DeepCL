#include "EasyCL.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

#include <sstream>
#include <iostream>
#include <string>

#include <clBLAS.h>

#include "clblas/ClBlasInstance.h"
#include "BackpropWeightsIm2Col.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

#define PUBLIC

PUBLIC BackpropWeightsIm2Col::BackpropWeightsIm2Col(EasyCL *cl, LayerDimensions dim) :
            BackpropWeights(cl, dim)
        {
    ClBlasInstance::initializeIfNecessary();

//    addBias = new AddBias(cl);

    int size = dim.inputSize;
    int padding = dim.padZeros ? dim.halfFilterSize : 0;
    int stride = 1;
    int channels = dim.inputPlanes;
    int size_col = (size + 2 * padding - dim.filterSize) / stride + 1;

    this->numKernels = channels * dim.inputSizeSquared;

    TemplatedKernel builder(cl);
    builder.set("padding", dim.padZeros ? dim.halfFilterSize : 0);
    builder.set("stride", 1);
    builder.set("colSize", size_col);
    builder.set("channels", dim.inputPlanes);
    builder.set("filterSize", dim.filterSize);
    builder.set("size", dim.inputSize);
    this->kernelIm2Col = builder.buildKernel(
        "im2col",
        "ForwardIm2Col.cl",
        getKernelTemplate(),
        "im2col",
        false);
}
PUBLIC VIRTUAL BackpropWeightsIm2Col::~BackpropWeightsIm2Col() {
    delete kernelIm2Col;
//    delete addBias;
}
//int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper
PUBLIC VIRTUAL void BackpropWeightsIm2Col::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *inputWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights START");

    int columnsSize = dim.inputPlanes * dim.filterSizeSquared * dim.outputSizeSquared;
    float *columns = new float[columnsSize];
    CLWrapper *gradColumnsWrapper = cl->wrap(columnsSize, columns);
    columnsWrapper->createOnDevice();
//    cout << "gradColumnsSize: " << gradColumnsSize << endl;
//    cout << "weightsize: " << weightsWrapper->size() << endl;

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights after alloc");

    
    for (int b = 0; b < batchSize; b ++) {
//        cout << "b=" << b << " numkernels=" << numKernels << endl;

        im2col(
          inputWrapper, b * dim.inputCubeSize,
          columnsWrapper
        );

        ClBlasHelper::Gemm(
            cl, clblasColumnMajor, clblasTrans, clblasNoTrans,
            m, n, k,
            scale,
            columnsWrapper, 0,
            gradOutputWrapper, b * dim.outputCubeSize,
            1,
            gradWeightsWrapper, 0
        );

        // Do Bias:
        long n_ = dim.numFilters;
        long m_ = dim.outputSizeSquared;

        // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
        ClBlasHelper:Gemv(
            clblasColumnMajor,
            clblasTrans,
            m_, n_,
            scale,
            gradOutputWrapper, b * dim.outputCubeSize,
            onesWrapper, 0,
            1,
            gradBiasWrapper, 0
        );
    }

    delete columnsWrapper;
    delete columns;

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights after call calcGradWeights");

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights END");
}
STATIC std::string BackpropWeightsIm2Col::getKernelTemplate() {
    // [[[cog
    // import stringify
    // stringify.write_kernel("kernel", "cl/ForwardIm2Col.cl")
    // ]]]
    // generated using cog, from cl/ForwardIm2Col.cl:
    const char * kernelSource =  
    "// from SpatialConvolutionMM.cu:\n" 
    "\n" 
    "// CL: grid stride looping\n" 
    "#define CL_KERNEL_LOOP(i, n)                        \\\n" 
    "  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \\\n" 
    "      i < (n);                                       \\\n" 
    "      i += get_local_size(0) * get_num_groups(0))\n" 
    "\n" 
    "//#define gPadding {{padding}}\n" 
    "//#define gStride {{stride}}\n" 
    "//#define gColSize {{colSize}}\n" 
    "//#define gFilterSize {{filterSize}}\n" 
    "//#define gSize {{size}}\n" 
    "\n" 
    "// Kernel for fast unfold+copy\n" 
    "// (adapted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)\n" 
    "kernel void im2col(\n" 
    "    const int n,\n" 
    "    global float const * im_data, int im_offset,\n" 
    "    global float* data_col) {\n" 
    "  global const float *data_im = im_data + im_offset;\n" 
    "\n" 
    "  CL_KERNEL_LOOP(index, n) {\n" 
    "    int w_out = index % {{colSize}};\n" 
    "    index /= {{colSize}};\n" 
    "    int h_out = index % {{colSize}};\n" 
    "    int channel_in = index / {{colSize}};\n" 
    "    int channel_out = channel_in * {{filterSize}} * {{filterSize}};\n" 
    "    int h_in = h_out * {{stride}} - {{padding}};\n" 
    "    int w_in = w_out * {{stride}} - {{padding}};\n" 
    "    data_col += (channel_out * {{colSize}} + h_out) * {{colSize}} + w_out;\n" 
    "    data_im += (channel_in * {{size}} + h_in) * {{size}} + w_in;\n" 
    "    for (int i = 0; i < {{filterSize}}; ++i) {\n" 
    "      for (int j = 0; j < {{filterSize}}; ++j) {\n" 
    "        int h = h_in + i;\n" 
    "        int w = w_in + j;\n" 
    "        *data_col = (h >= 0 && w >= 0 && h < {{size}} && w < {{size}}) ?\n" 
    "          data_im[i * {{size}} + j] : 0;\n" 
    "        data_col += {{colSize}} * {{colSize}};\n" 
    "      }\n" 
    "    }\n" 
    "  }\n" 
    "}\n" 
    "\n" 
    "kernel void col2im(\n" 
    "    const int n,\n" 
    "    global float const *data_col,\n" 
    "    global float* im_data, int im_offset) {\n" 
    "  global float *data_im = im_data + im_offset;\n" 
    "\n" 
    "  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0); index < (n); index += get_local_size(0) * get_num_groups(0)) {\n" 
    "    float val = 0;\n" 
    "    int w = index % {{size}} + {{padding}};\n" 
    "    int h = (index / {{size}}) % {{size}} + {{padding}};\n" 
    "    int c = index / ({{size}} * {{size}});\n" 
    "    // compute the start and end of the output\n" 
    "    int w_col_start = (w < {{filterSize}}) ? 0 : (w - {{filterSize}}) / {{stride}} + 1;\n" 
    "    int w_col_end = min(w / {{stride}} + 1, {{colSize}});\n" 
    "    int h_col_start = (h < {{filterSize}}) ? 0 : (h - {{filterSize}}) / {{stride}} + 1;\n" 
    "    int h_col_end = min(h / {{stride}} + 1, {{colSize}});\n" 
    "\n" 
    "    int offset = (c * {{filterSize}} * {{filterSize}} + h * {{filterSize}} + w) * {{colSize}} * {{colSize}};\n" 
    "    int coeff_h_col = (1 - {{stride}} * {{filterSize}} * {{colSize}}) * {{colSize}};\n" 
    "    int coeff_w_col = (1 - {{stride}} * {{colSize}} * {{colSize}});\n" 
    "    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n" 
    "      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n" 
    "        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n" 
    "      }\n" 
    "    }\n" 
    "    data_im[index] = val;\n" 
    "  }\n" 
    "}\n" 
    "\n" 
    "";
    // [[[end]]]
    return kernelSource;
}

