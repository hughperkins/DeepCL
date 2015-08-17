// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>

#include "conv/Forward.h"
#include "util/stringhelper.h"
#include "conv/ForwardCpu.h"
#include "conv/Forward1.h"
#include "conv/Forward2.h"
#include "conv/Forward3.h"
#include "conv/Forward4.h"
#include "conv/ForwardFc.h"
#include "conv/ForwardByInputPlane.h"
#include "conv/ForwardIm2Col.h"
#include "conv/ForwardAuto.h"
#include "util/StatefulTimer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

Forward::Forward(EasyCL *cl, LayerDimensions layerDimensions) :
        cl(cl),
        dim(layerDimensions) {
}
STATIC Forward *Forward::instance(EasyCL *cl, LayerDimensions dim) {
    return new ForwardAuto(cl, dim);
//    return new ForwardByInputPlane(cl, dim);

//    if(dim.filterSize == dim.inputSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11) {
//        return new ForwardFc(cl, dim);
//    } else {
//    }
//    if(dim.filterSize == dim.inputSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11) {
//        return new ForwardFc(cl, dim);
//    } else if(square(dim.outputSize) < 32 || square(dim.outputSize) > cl->getMaxWorkgroupSize()) {
//        return new Forward1(cl, dim);
//    } else {
//        return new Forward3(cl, dim);
//    }
}
STATIC Forward *Forward::instanceTest(EasyCL *cl, LayerDimensions layerDimensions) {
    return new Forward2(cl, layerDimensions);
}
STATIC int Forward::getNumImplementations() {
    return 8;
}
STATIC bool Forward::plausiblyOptimal(int index, int batchSize, LayerDimensions dim) {
    if(index == 0) { 
        return false;
    }
    if(index > 7) {
        return false;
    }
    return true;
}
STATIC Forward *Forward::instanceSpecific(int idx, EasyCL *cl, LayerDimensions layerDimensions) {
    if(idx == 0) {
        return new ForwardCpu(cl, layerDimensions);
    } else if(idx == -1) {
        return instance(cl, layerDimensions);
    } else if(idx == -2) {
        cout << "Forward::instanceSpeicfic, choosing: ForwardAuto" << endl;
        return new ForwardAuto(cl, layerDimensions);
    } else if(idx == 1) {
        return new Forward1(cl, layerDimensions);
    } else if(idx == 2) {
        return new Forward2(cl, layerDimensions);
    } else if(idx == 3) {
        return new Forward3(cl, layerDimensions);
    } else if(idx == 4) {
        return new Forward4(cl, layerDimensions);
    } else if(idx == 5) {
        return new ForwardFc(cl, layerDimensions);
    } else if(idx == 6) {
        return new ForwardByInputPlane(cl, layerDimensions);
    } else if(idx == 7) {
        return new ForwardIm2Col(cl, layerDimensions);
    } else {
        throw runtime_error(string("") + __FILE__ + ":" + toString(__LINE__) + " Forward::instanceSpecific: no instance defined for index " + toString(idx));
    }
}
STATIC Forward *Forward::instanceSpecific(std::string name, EasyCL *cl, LayerDimensions layerDimensions) {
    if(name == "cpu") {
        return new ForwardCpu(cl, layerDimensions);
    } else if(name == "prop1") {
        return new Forward1(cl, layerDimensions);
    } else if(name == "prop3") {
        return new Forward3(cl, layerDimensions);
    } else if(name == "prop4") {
        return new Forward4(cl, layerDimensions);
    } else if(name == "fc") {
        return new ForwardFc(cl, layerDimensions);
    } else if(name == "byinplane") {
        return new ForwardByInputPlane(cl, layerDimensions);
    } else {
        throw runtime_error(string("") + __FILE__ + ":" + toString(__LINE__) + " Forward::instanceSpecific: no instance defined for name " + name);
    }
}
// you own the returned output array, and are responsible for deleting it
//VIRTUAL float * Forward::forward(int batchSize, float *inputData, float *filters, float *biases) {
//    float *output = new float[batchSize * dim.outputCubeSize];
//    forward(batchSize, inputData, filters, biases, output);
//    return output;
//}
VIRTUAL int Forward::getOutputTotalSize(int batchSize) {
    return batchSize * dim.outputCubeSize;
}
// must allocate output yourself before the call
VIRTUAL void Forward::forward(int batchSize, float *inputData, float *filters, float *biases, float *output) {
    StatefulTimer::timeCheck("Forward::forward begin");
    int inputDataSize = batchSize * dim.inputCubeSize;
    CLWrapper *dataWrapper = cl->wrap(inputDataSize, inputData);
    dataWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = cl->wrap(weightsSize, filters);
    weightsWrapper->copyToDevice();

    CLWrapper *biasWrapper = 0;
    if(dim.biased) {
        int biasWrapperSize = dim.numFilters;
        biasWrapper = cl->wrap(biasWrapperSize, biases);
        biasWrapper->copyToDevice();
    }

//    int outputDataSize = batchSize * dim.outputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
//    int allocatedOutputNumElements = std::max(5000, outputDataSize);
//    int allocatedOutputNumElements = outputDataSize;
//    float *output = new float[allocatedOutputNumElements];
    CLWrapper *outputWrapper = cl->wrap(batchSize * dim.outputCubeSize, output);
    outputWrapper->createOnDevice();
    cl->finish();

    StatefulTimer::timeCheck("Forward::forward after copied to device");
    forward(batchSize, dataWrapper, weightsWrapper, biasWrapper,
            outputWrapper);
    StatefulTimer::timeCheck("Forward::forward after call forward");
    cl->finish();
    outputWrapper->copyToHost();
    StatefulTimer::timeCheck("Forward::forward after copytohost");
//    for(int i = 0; i < 20; i++) {
//        cout << "output[" << i << "]=" << output[i] << endl;
//    }
    delete outputWrapper;

    delete dataWrapper;
    delete weightsWrapper;
    if(dim.biased) {
        delete biasWrapper;
    }

//    return output;
}

