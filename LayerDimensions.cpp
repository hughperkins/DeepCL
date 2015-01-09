#include <iostream>
#include <ostream>

#include "LayerDimensions.h"

using namespace std;

ostream &operator<<( ostream &os, const LayerDimensions &dim ) {
    os << "LayerDimensions{";
    os << " inputPlanes=" << dim.inputPlanes;
    os << " inputBoardSize=" << dim.inputBoardSize;
    os << " numFilters=" << dim.numFilters;
    os << " filterSize=" << dim.filterSize;
    os << " outputBoardSize=" << dim.outputBoardSize;
    os << " padZeros=" << dim.padZeros;
    os << " biased=" << dim.biased;
    os << "}";
    return os;
}


