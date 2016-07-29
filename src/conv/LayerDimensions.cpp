#include <iostream>
#include <ostream>

#include "util/stringhelper.h"

#include "conv/LayerDimensions.h"

using namespace std;

ostream &operator<<(ostream &os, const LayerDimensions &dim) {
    os << "LayerDimensions{";
    os << " inputPlanes=" << dim.inputPlanes;
    os << " iH=" << dim.iH << " iW=" << dim.iW;
    os << " numFilters=" << dim.numFilters;
    os << " filterSize=" << dim.filterSize;
    os << " oH=" << dim.oH << " oW=" << dim.oW;
    os << " padZeros=" << dim.padZeros;
    os << " biased=" << dim.biased;
    os << " skip=" << dim.skip;
    os << "}";
    return os;
}

void LayerDimensions::deriveOthers() {
    this->numInputPlanes = inputPlanes;
    this->isEven = filterSize % 2 == 0;
    this->oH = padZeros ? 
            (filterSize % 2 == 0 ? iH / (skip + 1) + 1 : iH / (skip + 1) ) :
            (iH - filterSize) / (skip + 1) + 1;
    this->oW = padZeros ? 
            (filterSize % 2 == 0 ? iW / (skip + 1) + 1 : iW / (skip + 1) ) :
            (iW - filterSize) / (skip + 1) + 1;

    this->inputSizeSquared = iH * iW;
    this->filterSizeSquared = filterSize * filterSize;
    this->outputSizeSquared = oH * oW;

    this->inputCubeSize = inputPlanes * inputSizeSquared;
    this->filtersSize = inputPlanes * numFilters * filterSizeSquared;
    this->outputCubeSize = numFilters * outputSizeSquared;

    this->halfFilterSize = filterSize >> 1;
//    cout << "deriveOthers()" << *this << endl;
}

string LayerDimensions::buildOptionsString() {
    string options = "";
    if(biased) {
         options += " -D BIASED";
    }
    options += " -D gNumInputPlanes=" + toString(inputPlanes);
    options += " -D gInputPlanes=" + toString(inputPlanes);
    options += " -D gInputSize=" + toString(inputSize);
    options += " -D gIH=" + toString(iH);
    options += " -D gIW=" + toString(iW);
    options += " -D gInputSizeSquared=" + toString(square(inputSize));
    options += " -D gNumFilters=" + toString(numFilters);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gHalfFilterSize=" + toString(filterSize >> 1);
    options += " -D gFilterSizeSquared=" + toString(square(filterSize));
    options += " -D gNumOutputPlanes=" + toString(numFilters);
    options += " -D gOutputPlanes=" + toString(numFilters);
    options += " -D gOutputSize=" + toString(outputSize);
    options += " -D gOutputSizeSquared=" + toString(square(outputSize));
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gEven=" + toString(filterSize % 2 == 0 ? 1 : 0);
    options += " -D gSkip=" + toString(skip);
    return options;
}

