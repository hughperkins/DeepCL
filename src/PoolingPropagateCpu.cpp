// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"

#include "PoolingPropagateCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingPropagateCpu::PoolingPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) :
        PoolingPropagate( cl, numPlanes, inputBoardSize, poolingSize ) {
}
VIRTUAL void PoolingPropagateCpu::propagate( int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper ) {
    float *input = reinterpret_cast<float *>( inputWrapper->getHostArray() );

    float *output = propagate( batchSize, input );

    float *outputHostArray = reinterpret_cast<float *>( outputWrapper->getHostArray() );
    memcpy( outputHostArray, output, sizeof(float) * getResultsSize( batchSize ) );
    delete[] output;
}
VIRTUAL float *PoolingPropagateCpu::propagate( int batchSize, float *input ) {
    float *output = new float[ getResultsSize( batchSize ) ];
    for( int n = 0; n < batchSize; n++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outputRow = 0; outputRow < outputBoardSize; outputRow++ ) {
                int inputRow = outputRow * poolingSize;
                for( int outputCol = 0; outputCol < outputBoardSize; outputCol++ ) {
                    int inputCol = outputCol * poolingSize;
                    float maxValue = input[ getInputIndex( n, plane, inputRow, inputCol ) ];
                    for( int dx = 0; dx < poolingSize; dx++ ) {
                        for( int dy = 0; dy < poolingSize; dy++ ) {
                            float thisValue = input[ getInputIndex( n, plane, inputRow + dx, inputCol + dy ) ];
                            maxValue = std::max( maxValue, thisValue );
                        }
                    }
                    output[getResultIndex( n, plane, outputRow, outputCol ) ] = maxValue;
                }
            }
        }
    }
    return output;
}

