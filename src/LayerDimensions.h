#pragma once

#include <iostream>
#include <cstring>

#include "DllImportExport.h"

inline int square( int value ) {
    return value * value;
}

class ClConvolve_EXPORT LayerDimensions {
public:
    int inputPlanes, inputBoardSize, numFilters, filterSize, outputBoardSize;
    bool padZeros, isEven;
    bool biased;
    int skip;

    int inputCubeSize;
    int filtersSize;
    int outputCubeSize;

    int halfFilterSize;

    LayerDimensions() {
        memset( this, 0, sizeof( LayerDimensions ) );
    }
    LayerDimensions( int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                bool padZeros, bool biased ) :
            inputPlanes( inputPlanes ),
            inputBoardSize( inputBoardSize ),
            numFilters( numFilters ),
            filterSize( filterSize ),
            padZeros( padZeros ),
            biased( biased )
        {
        skip = 0;
        deriveOthers();
//        std::cout << "outputBoardSize " << outputBoardSize << " padZeros " << padZeros << " filtersize "
//            << filterSize << " inputBoardSize " << inputBoardSize << std::endl;
    }
    LayerDimensions &setInputPlanes( int _planes ) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setInputBoardSize( int inputBoardSize ) {
        this->inputBoardSize = inputBoardSize;
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
    void deriveOthers() {
        this->isEven = filterSize % 2 == 0;
        outputBoardSize = padZeros ? 
                ( filterSize % 2 == 0 ? inputBoardSize / ( skip + 1 ) + 1 : inputBoardSize / ( skip + 1 ) ) :
                ( inputBoardSize - filterSize ) / ( skip + 1 ) + 1;
        inputCubeSize = inputPlanes * inputBoardSize * inputBoardSize;
        filtersSize = inputPlanes * numFilters * filterSize * filterSize;
        outputCubeSize = numFilters * outputBoardSize * outputBoardSize;

        halfFilterSize = filterSize >> 1;
    }
    std::string buildOptionsString();
};

ClConvolve_EXPORT std::ostream &operator<<( std::ostream &os, const LayerDimensions &dim );


