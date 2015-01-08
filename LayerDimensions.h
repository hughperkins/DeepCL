#pragma once

inline int square( int value ) {
    return value * value;
}

class LayerDimensions {
public:
    const int inputPlanes, inputBoardSize, numFilters, filterSize, outputBoardSize;
    const bool padZeros, isEven;
    const bool biased;

    LayerDimensions( int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                bool padZeros, bool biased ) :
            inputPlanes( inputPlanes ),
            inputBoardSize( inputBoardSize ),
            numFilters( numFilters ),
            filterSize( filterSize ),
            padZeros( padZeros ),
            biased( biased ),
            isEven( filterSize % 2 == 0 ),
            outputBoardSize( padZeros ? 
                ( filterSize % 2 == 0 ? inputBoardSize + 1 : inputBoardSize ) :
                inputBoardSize - filterSize + 1 )
        {
    }    
};



