#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

#include <sstream>
#include <iostream>
#include <string>

//#include "clblas/ClBlasInstance.h"
#include "clblas/ClBlasHelper.h"
#include "conv/Im2Col.h"
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
//    ClBlasInstance::initializeIfNecessary();
    im2Col = new Im2Col(cl, dim);
}
PUBLIC VIRTUAL BackwardIm2Col::~BackwardIm2Col() {
    delete im2Col;
}
PUBLIC VIRTUAL void BackwardIm2Col::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {
    StatefulTimer::timeCheck("BackwardIm2Col::backward START");

    int gradColumnsSize = dim.inputPlanes * dim.filterSizeSquared * dim.outputSizeSquared;
    float *gradColumns = new float[gradColumnsSize];
    CLWrapper *gradColumnsWrapper = cl->wrap(gradColumnsSize, gradColumns);
    gradColumnsWrapper->createOnDevice();
//    cout << "gradColumnsSize: " << gradColumnsSize << endl;
//    cout << "weightsize: " << weightsWrapper->size() << endl;

    StatefulTimer::timeCheck("BackwardIm2Col::backward after alloc");

    if(!gradInputWrapper->isOnDevice()) {
        gradInputWrapper->createOnDevice();
    }
    for (int b = 0; b < batchSize; b ++) {
//        cout << "b=" << b << " numkernels=" << numKernels << endl;
        long m = dim.outputSizeSquared;
        long n = dim.inputPlanes * dim.filterSizeSquared;
        long k = dim.numFilters;
//        cout << "m=" << m << " k=" << k << " n=" << n << endl;

        ClBlasHelper::Gemm(
            cl, clblasColumnMajor, clblasNoTrans, clblasTrans,
            m, k, n,
            1,
            gradOutputWrapper, b * dim.outputCubeSize,
            weightsWrapper, 0,
            0,
            gradColumnsWrapper, 0
        );

        im2Col->col2Im(gradColumnsWrapper, gradInputWrapper, b * dim.inputCubeSize);
    }

    delete gradColumnsWrapper;
    delete[] gradColumns;

    StatefulTimer::timeCheck("BackwardIm2Col::backward after call backward");

    StatefulTimer::timeCheck("BackwardIm2Col::backward END");
}

