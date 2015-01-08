#pragma once

#include "Propagate.h"

class PropagateCpu : public Propagate {
public:
    PropagateCpu( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction *fn ) :
            Propagate( cl, dim, fn )
        {
    }
    virtual float *propagate( int batchSize, float *inputData, float *weights, float *biasWeights ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int filter = 0; filter < dim.numFilters; filter++ ) {
                for( int inPlane = 0; inPlane < dim.inputPlanes; inPlane++ ) {
                    for( int outRow = 0; outRow < dim.outputBoardSize; outRow++ ) {
                        for( int outCol = 0; outCol < dim.outputBoardSize; outCol++ ) {
                            int minu = dim.padZeros ? std::max( -halfFilterSize, -outRow ) : -dim.halfFilterSize;
                            int maxu = dim.padZeros ? std::min( dim.halfFilterSize - dim.evenPadding, dim.outputBoardSize - 1 - outRow  - dim.evenPadding) : dim.halfFilterSize - dim.evenPadding;
                            int minv = dim.padZeros ? std::max( -dim.halfFilterSize, -outCol ) : - dim.halfFilterSize;
                            int maxv = dim.padZeros ? std::min( dim.halfFilterSize - dim.evenPadding, dim.outputBoardSize - 1 - outCol - dim.evenPadding) : dim.halfFilterSize - dim.evenPadding;
                            float sum = 0;
                            for( int u = minu; u <= maxu; u++ ) {
                                int inRow = outRow + u + ( dim.padZeros ? 0 : dim.halfFilterSize );
                                for( int v = minv; v <= maxv; v++ ) {
                                    int inCol = outCol + v + ( dim.padZeros ? 0 : dim.halfFilterSize );
//                                    int dataIndex = 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

