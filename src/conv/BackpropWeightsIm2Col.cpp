#include "EasyCL.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

#include <sstream>
#include <iostream>
#include <string>

//#include "clblas/ClBlasInstance.h"
#include "clblas/ClBlasHelper.h"
#include "BackpropWeightsIm2Col.h"
#include "conv/Im2Col.h"
#include "clmath/CLMathWrapper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

#define PUBLIC

PUBLIC BackpropWeightsIm2Col::BackpropWeightsIm2Col(EasyCL *cl, LayerDimensions dim) :
            BackpropWeights(cl, dim)
        {
//    ClBlasInstance::initializeIfNecessary();

//    addBias = new AddBias(cl);

    this->im2Col = new Im2Col(cl, dim);
}
PUBLIC VIRTUAL BackpropWeightsIm2Col::~BackpropWeightsIm2Col() {
    delete im2Col;
//    delete addBias;
}
//int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper
PUBLIC VIRTUAL void BackpropWeightsIm2Col::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *inputWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights START");

    int columnsSize = dim.inputPlanes * dim.filterSizeSquared * dim.outputSizeSquared;
    float *columns = new float[columnsSize];
    CLWrapper *columnsWrapper = cl->wrap(columnsSize, columns);
    columnsWrapper->createOnDevice();

    int onesSize = dim.outputSizeSquared;
    float *ones = new float[onesSize];
    CLWrapper *onesWrapper = cl->wrap(onesSize, ones);
    onesWrapper->createOnDevice();
    CLMathWrapper ones_(onesWrapper);
    ones_ = 1.0f;

//    cout << "gradColumnsSize: " << gradColumnsSize << endl;
//    cout << "weightsize: " << weightsWrapper->size() << endl;

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights after alloc");

    CLMathWrapper gradWeights_(gradWeightsWrapper);
    gradWeights_ = 0.0f;
    if(dim.biased) {
        CLMathWrapper gradBias_(gradBiasWrapper);
        gradBias_ = 0.0f;
    }
    for (int b = 0; b < batchSize; b ++) {
//        cout << "b=" << b << " numkernels=" << numKernels << endl;

        im2Col->im2Col(
            inputWrapper, b * dim.inputCubeSize,
            columnsWrapper
        );
        
        int64 m = dim.inputPlanes * dim.filterSizeSquared;
        int64 n = dim.numFilters;
        int64 k = dim.outputSizeSquared;

        ClBlasHelper::Gemm(
            cl,
            clblasColumnMajor,
            clblasTrans, clblasNoTrans,
            m, k, n,
            1,
            columnsWrapper, 0,
            gradOutputWrapper, b * dim.outputCubeSize,
            1,
            gradWeightsWrapper, 0
        );
        if(dim.biased) {
            int64 m_ = dim.outputSizeSquared;
            int64 n_ = dim.numFilters;
            ClBlasHelper::Gemv(
                cl,
                clblasColumnMajor,
                clblasTrans,
                m_, n_,
                1,
                gradOutputWrapper, b * dim.outputCubeSize,
                onesWrapper, 0,
                1,
                gradBiasWrapper, 0
            );
        }
    }

    delete onesWrapper;
    delete[] ones;

    delete columnsWrapper;
    delete[] columns;

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights after call calcGradWeights");

    StatefulTimer::timeCheck("BackpropWeightsIm2Col::calcGradWeights END");
}

