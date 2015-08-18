// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ForwardIm2Col.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "conv/AddBias.h"
//#include "clblas/ClBlasInstance.h"
#include "clblas/ClBlasHelper.h"
#include "conv/Im2Col.h"

#include <sstream>
#include <iostream>
#include <string>

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
#define PUBLIC

PUBLIC ForwardIm2Col::ForwardIm2Col(EasyCL *cl, LayerDimensions dim) :
            Forward(cl, dim)
        {
//    ClBlasInstance::initializeIfNecessary();

    addBias = new AddBias(cl);
    im2Col = new Im2Col(cl, dim);
}
PUBLIC VIRTUAL ForwardIm2Col::~ForwardIm2Col() {
    delete addBias;
    delete im2Col;
}
PUBLIC VIRTUAL void ForwardIm2Col::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper) {
    StatefulTimer::timeCheck("ForwardIm2Col::forward START");

    int columnsSize= dim.inputPlanes * dim.filterSizeSquared * dim.outputSizeSquared;
    float *columns = new float[columnsSize];
    CLWrapper *columnsWrapper = cl->wrap(columnsSize, columns);
    columnsWrapper->createOnDevice();
//    cout << "columnsSize: " << columnsSize << endl;
//    cout << "weightsize: " << weightsWrapper->size() << endl;

    StatefulTimer::timeCheck("ForwardIm2Col::forward after alloc");

    for (int b = 0; b < batchSize; b ++) {
        im2Col->im2Col(dataWrapper, b * dim.inputCubeSize, columnsWrapper);

        long m = dim.outputSizeSquared;
        long n = dim.numFilters;
        long k = dim.inputPlanes * dim.filterSizeSquared;
//        cout << "m=" << m << " n=" << n << " k=" << k << endl;

        ClBlasHelper::Gemm(
            cl, clblasColumnMajor, clblasNoTrans, clblasNoTrans,
            m, k, n,
            1,
            columnsWrapper, 0,
            weightsWrapper, 0,
            0,
            outputWrapper, b * dim.outputCubeSize
        );
    }

    delete columnsWrapper;
    delete[] columns;

    StatefulTimer::timeCheck("ForwardIm2Col::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("ForwardIm2Col::forward END");
}

