#pragma once

#include <iostream>
#include <cstring>

#include "DeepCLDllExport.h"

inline int square( int value ) {
    return value * value;
}

class DeepCL_EXPORT LayerDimensions {
public:
    int inputPlanes, inputImageSize, numFilters, filterSize, outputImageSize;
    bool padZeros, isEven;
    bool biased;
    int skip;

    int inputCubeSize;
    int filtersSize;
    int outputCubeSize;
    int numInputPlanes;

    int outputImageSizeSquared;
    int filterSizeSquared;
    int inputImageSizeSquared;

    int halfFilterSize;

    LayerDimensions() {
        memset( this, 0, sizeof( LayerDimensions ) );
    }
    LayerDimensions( int inputPlanes, int inputImageSize, 
                int numFilters, int filterSize, 
                bool padZeros, bool biased ) :
            inputPlanes( inputPlanes ),
            inputImageSize( inputImageSize ),
            numFilters( numFilters ),
            filterSize( filterSize ),
            padZeros( padZeros ),
            biased( biased )
        {
        skip = 0;
        deriveOthers();
//        std::cout << "outputImageSize " << outputImageSize << " padZeros " << padZeros << " filtersize "
//            << filterSize << " inputImageSize " << inputImageSize << std::endl;
    }
    LayerDimensions &setInputPlanes( int _planes ) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumInputPlanes( int _planes ) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setInputImageSize( int inputImageSize ) {
        this->inputImageSize = inputImageSize;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setSkip( int skip ) {
        this->skip = skip;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumFilters( int numFilters ) {
        this->numFilters = numFilters;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setFilterSize( int filterSize ) {
        this->filterSize = filterSize;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setBiased( bool biased ) {
        this->biased = biased;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setPadZeros( bool padZeros ) {
        this->padZeros = padZeros;
        deriveOthers();
        return *this;
    }
    void deriveOthers();
    std::string buildOptionsString();
};

DeepCL_EXPORT std::ostream &operator<<( std::ostream &os, const LayerDimensions &dim );


