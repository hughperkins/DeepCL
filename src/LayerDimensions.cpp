#include <iostream>
#include <ostream>

#include "stringhelper.h"

#include "LayerDimensions.h"

using namespace std;

ostream &operator<<( ostream &os, const LayerDimensions &dim ) {
    os << "LayerDimensions{";
    os << " inputPlanes=" << dim.inputPlanes;
    os << " inputImageSize=" << dim.inputImageSize;
    os << " numFilters=" << dim.numFilters;
    os << " filterSize=" << dim.filterSize;
    os << " outputImageSize=" << dim.outputImageSize;
    os << " padZeros=" << dim.padZeros;
    os << " biased=" << dim.biased;
    os << " skip=" << dim.skip;
    os << "}";
    return os;
}

void LayerDimensions::deriveOthers() {
    this->numInputPlanes = inputPlanes;
    this->isEven = filterSize % 2 == 0;
    this->outputImageSize = padZeros ? 
            ( filterSize % 2 == 0 ? inputImageSize / ( skip + 1 ) + 1 : inputImageSize / ( skip + 1 ) ) :
            ( inputImageSize - filterSize ) / ( skip + 1 ) + 1;

    this->inputImageSizeSquared = inputImageSize * inputImageSize;
    this->filterSizeSquared = filterSize * filterSize;
    this->outputImageSizeSquared = outputImageSize * outputImageSize;

    this->inputCubeSize = inputPlanes * inputImageSizeSquared;
    this->filtersSize = inputPlanes * numFilters * filterSizeSquared;
    this->outputCubeSize = numFilters * outputImageSizeSquared;

    this->halfFilterSize = filterSize >> 1;
//    cout << "deriveOthers()" << *this << endl;
}

string LayerDimensions::buildOptionsString() {
    string options = "";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gNumInputPlanes=" + toString(inputPlanes);
    options += " -D gInputPlanes=" + toString(inputPlanes);
    options += " -D gInputImageSize=" + toString(inputImageSize);
    options += " -D gInputImageSizeSquared=" + toString(square(inputImageSize));
    options += " -D gNumFilters=" + toString(numFilters);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gFilterSizeSquared=" + toString(square(filterSize));
    options += " -D gNumOutputPlanes=" + toString(numFilters);
    options += " -D gOutputPlanes=" + toString(numFilters);
    options += " -D gOutputImageSize=" + toString(outputImageSize);
    options += " -D gOutputImageSizeSquared=" + toString(square(outputImageSize));
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gEven=" + toString(filterSize % 2 == 0 ? 1 : 0);
    options += " -D gSkip=" + toString(skip);
    return options;
}

