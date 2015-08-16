#include "EasyCL.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

#include <sstream>
#include <iostream>
#include <string>

#include <clBLAS.h>

#include "BackwardIm2Col.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

#define PUBLIC

PUBLIC BackwardIm2Col::BackwardIm2Col(EasyCL *cl, LayerDimensions dim) :
            Backward(cl, dim)
        {
//    addBias = new AddBias(cl);

    int size = dim.inputSize;
    int padding = dim.padZeros ? dim.halfFilterSize : 0;
    int stride = 1;
    int channels = dim.inputPlanes;
    int size_col = (size + 2 * padding - dim.filterSize) / stride + 1;

    this->numKernels = channels * size_col * size_col;

    TemplatedKernel builder(cl);
    builder.set("padding", dim.padZeros ? dim.halfFilterSize : 0);
    builder.set("stride", 1);
    builder.set("colSize", size_col);
    builder.set("channels", dim.inputPlanes);
    builder.set("filterSize", dim.filterSize);
    builder.set("size", dim.inputSize);
    this->kernelCol2Im = builder.buildKernel(
        "col2im",
        "ForwardIm2Col.cl",
        getKernelTemplate(),
        "col2im",
        false);
}
PUBLIC VIRTUAL BackwardIm2Col::~BackwardIm2Col() {
    delete kernelCol2Im;
//    delete addBias;
}
PUBLIC VIRTUAL void BackwardIm2Col::backward( int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper ) {
    StatefulTimer::timeCheck("BackwardIm2Col::backward START");

    int gradColumnsSize = dim.inputPlanes * dim.filterSizeSquared * dim.outputSizeSquared;
    float *gradColumns = new float[gradColumnsSize];
    CLWrapper *gradColumnsWrapper = cl->wrap(gradColumnsSize, gradColumns);
//    cout << "columnsSize: " << columnsSize << endl;
//    cout << "weightsize: " << weightsWrapper->size() << endl;

    StatefulTimer::timeCheck("BackwardIm2Col::backward after alloc");

    for (int b = 0; b < batchSize; b ++) {
//        cout << "b=" << b << " numkernels=" << numKernels << endl;
        // Extract columns:

        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
        long m = dim.outputSizeSquared;
        long n = dim.numFilters;
        long k = dim.inputPlanes * dim.filterSizeSquared;
//        cout << "m=" << m << " n=" << n << " k=" << k << endl;

        clblasOrder order = clblasColumnMajor;
        size_t lda = order == clblasRowMajor ? k : m;
        size_t ldb = order == clblasRowMajor ? n : k;
        size_t ldc = order == clblasRowMajor ? n : m;
        cl_int err = clblasSgemm(
            order,
            clblasNoTrans, clblasTrans,
            m, n, k,
            1,
            gradOutputWrapper->getBuffer(), b * dim.outputCubeSize, lda,
            weightsWrapper->getBuffer(), 0, ldb,
            0,
            gradColumnsWrapper->getBuffer(), 0, ldc,
            1, cl->queue, 0, NULL, 0
       );
//    THClBlas_gemm(
//        'n', 't',
//        n, m, k,
//        1,
//        gradOutput_n, n,
//        weight, m,
//        0,
//        gradColumns, n
//    );
        if (err != CL_SUCCESS) {
            throw runtime_error("clblasSgemm() failed with " + toString(err));
        }

        kernelCol2Im->in(numKernels);
        kernelCol2Im->in(gradColumnsWrapper);
        kernelCol2Im->out(gradInputWrapper);
        kernelCol2Im->in(b * dim.inputCubeSize);

        int workgroupSize = cl->getMaxWorkgroupSize();
        int numWorkgroups = this->numKernels;

        kernelCol2Im->run_1d(numWorkgroups * workgroupSize, workgroupSize);

//        dataWrapper->copyToHost();
//        for( int i = 0; i < dataWrapper->size(); i++ ) {
//            cout << "data[" << i << "]=" << reinterpret_cast<float *>(dataWrapper->getHostArray())[i] << endl;
//        }
//        columnsWrapper->copyToHost();
//        for( int i = 0; i < columnsSize; i++ ) {
//            cout << "columns[" << i << "]=" << reinterpret_cast<float *>(columnsWrapper->getHostArray())[i] << endl;
//        }
//        weightsWrapper->copyToHost();
//        for( int i = 0; i < weightsWrapper->size(); i++ ) {
//            cout << "weights[" << i << "]=" << reinterpret_cast<float *>(weightsWrapper->getHostArray())[i] << endl;
//        }

//        cl->finish();
//        outputWrapper->copyToHost();
//        for( int i = 0; i < 1; i++ ) {
//            cout << "output[" << i << "]=" << reinterpret_cast<float *>(outputWrapper->getHostArray())[i] << endl;
//        }
    }

    delete gradColumnsWrapper;
    delete gradColumns;

    StatefulTimer::timeCheck("BackwardIm2Col::backward after call backward");

    StatefulTimer::timeCheck("BackwardIm2Col::backward END");
}
STATIC std::string BackwardIm2Col::getKernelTemplate() {
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
    "    global float const * col_data, int col_offset,\n" 
    "    global float* data_im) {\n" 
    "  global const float *data_col = col_data + col_offset;\n" 
    "\n" 
    "  CL_KERNEL_LOOP(index, n) {\n" 
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

