// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

#include "BackwardAuto.h"
#include "BackwardCpu.h"
#include "BackwardGpuNaive.h"
#include "BackwardGpuCached.h"
#include "BackwardIm2Col.h"

#include "Backward.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC Backward *Backward::instance(EasyCL *cl, LayerDimensions dim) {
    return new BackwardAuto(cl, dim);
//    if((dim.inputSize - dim.filterSize > 6) && square(dim.inputSize) <= cl->getMaxWorkgroupSize()) {
//        return new BackwardGpuCached(cl, dim);
//    } else {
//        return new BackwardGpuNaive(cl, dim);
//    }
}
STATIC Backward *Backward::instanceForTest(EasyCL *cl, LayerDimensions layerDimensions) {
    return new BackwardGpuNaive(cl, layerDimensions);
}
STATIC Backward *Backward::instanceSpecific(int idx, EasyCL *cl, LayerDimensions layerDimensions) {
    if(idx == -1) {
        return new BackwardAuto(cl, layerDimensions);
    }
    if(idx == 0) {
        return new BackwardCpu(cl, layerDimensions);
    }
    if(idx == 1) {
        return new BackwardGpuNaive(cl, layerDimensions);
    }
    if(idx == 2) {
        return new BackwardGpuCached(cl, layerDimensions);
    }
    if(idx == 3) {
        return new BackwardIm2Col(cl, layerDimensions);
    }
    throw std::runtime_error("backproperrorsv2::isntancespecifc, index not known: " + toString(idx));
}
Backward::Backward(EasyCL *cl, LayerDimensions layerDimensions) :
        cl(cl),
        dim(layerDimensions) {
}
STATIC int Backward::getNumImplementations() {
    return 4;
}
STATIC bool Backward::plausiblyOptimal(int index, int batchSize, LayerDimensions dim) {
    if(index == 0) { 
        return false;
    }
    if(index >= 4) {
        return false;
    }
    return true;
}
VIRTUAL float * Backward::backward(int batchSize, float *input, float *gradOutput, float *filters) {
    StatefulTimer::timeCheck("Backward::backprop begin");

    CLWrapper *inputWrapper = cl->wrap(batchSize * dim.inputCubeSize, input);
    inputWrapper->copyToDevice();

    CLWrapper *gradOutputWrapper = cl->wrap(batchSize * dim.outputCubeSize, gradOutput);
    gradOutputWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = cl->wrap(weightsSize, filters);
    weightsWrapper->copyToDevice();

    int outputDataSize = batchSize * dim.inputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
    int allocatedOutputNumElements = std::max(5000, outputDataSize);
    float *gradInput = new float[allocatedOutputNumElements];
    CLWrapper *gradInputWrapper = cl->wrap(allocatedOutputNumElements, gradInput);

    StatefulTimer::timeCheck("Backward::backprop after copied to device");
    backward(batchSize, inputWrapper, gradOutputWrapper, weightsWrapper, gradInputWrapper);
    StatefulTimer::timeCheck("Backward::backprop after call backprop");
    gradInputWrapper->copyToHost();
    StatefulTimer::timeCheck("Backward::backprop after copytohost");

    delete gradInputWrapper;
    delete gradOutputWrapper;
    delete weightsWrapper;
    delete inputWrapper;

    return gradInput;
}

