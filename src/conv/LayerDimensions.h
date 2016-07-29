#pragma once

#include <iostream>
#include <cstring>

#include "DeepCLDllExport.h"

inline int square(int value) {
    return value * value;
}

class DeepCL_EXPORT LayerDimensions {
public:
    int inputPlanes, iH, iW, numFilters, filterSize, oH, oW;
    bool padZeros, isEven;
    bool biased;
    int skip;

    int inputCubeSize;
    int filtersSize;
    int outputCubeSize;
    int numInputPlanes;

    int outputSizeSquared;
    int filterSizeSquared;
    int inputSizeSquared;

    int halfFilterSize;

    LayerDimensions() {
        memset(this, 0, sizeof(LayerDimensions) );
    }
    LayerDimensions(int inputPlanes, int iH, int iW, 
                int numFilters, int filterSize, 
                bool padZeros, bool biased) :
            inputPlanes(inputPlanes),
            iH(iH),
            iW(iW),
            numFilters(numFilters),
            filterSize(filterSize),
            padZeros(padZeros),
            biased(biased)
        {
        skip = 0;
        deriveOthers();
//        std::cout << "outputSize " << outputSize << " padZeros " << padZeros << " filtersize "
//            << filterSize << " inputSize " << inputSize << std::endl;
    }
    LayerDimensions &setInputPlanes(int _planes) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumInputPlanes(int _planes) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setIH(int iH) {
        this->iH = iH;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setSkip(int skip) {
        this->skip = skip;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumFilters(int numFilters) {
        this->numFilters = numFilters;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setFilterSize(int filterSize) {
        this->filterSize = filterSize;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setBiased(bool biased) {
        this->biased = biased;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setPadZeros(bool padZeros) {
        this->padZeros = padZeros;
        deriveOthers();
        return *this;
    }
    void deriveOthers();
    std::string buildOptionsString();
};

DeepCL_EXPORT std::ostream &operator<<(std::ostream &os, const LayerDimensions &dim);


