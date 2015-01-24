// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "OpenCLHelper.h"

#include "StatefulTimer.h"

#include "PoolingPropagateCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingPropagateCpu::PoolingPropagateCpu( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize ) :
        PoolingPropagate( cl, numPlanes, inputBoardSize, poolingSize ) {
}
VIRTUAL void PoolingPropagateCpu::propagate( int batchSize, CLWrapper *inputWrapper, CLWrapper *selectorsWrapper, CLWrapper *outputWrapper ) {
    inputWrapper->copyToHost();

    float *input = reinterpret_cast<float *>( inputWrapper->getHostArray() );
    int *selectors = new int[ getResultsSize( batchSize ) ];
    float *output = new float[ getResultsSize( batchSize ) ];

    propagate( batchSize, input, selectors, output );

    int *selectorsHostArray = reinterpret_cast<int *>( selectorsWrapper->getHostArray() );
    memcpy( selectorsHostArray, selectors, sizeof(int) * getResultsSize( batchSize ) );

    float *outputHostArray = reinterpret_cast<float *>( outputWrapper->getHostArray() );
    memcpy( outputHostArray, output, sizeof(float) * getResultsSize( batchSize ) );

    selectorsWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    delete[] selectors;
    delete[] output;
}
VIRTUAL void PoolingPropagateCpu::propagate( int batchSize, float *input, int *selectors, float *output ) {
//    float *output = new float[ getResultsSize( batchSize ) ];
    StatefulTimer::instance()->timeCheck("PoolingPropagateCpu::propagate start" );
    for( int n = 0; n < batchSize; n++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outputRow = 0; outputRow < outputBoardSize; outputRow++ ) {
                int inputRow = outputRow * poolingSize;
                for( int outputCol = 0; outputCol < outputBoardSize; outputCol++ ) {
                    int inputCol = outputCol * poolingSize;
                    int selector = 0;
                    float maxValue = input[ getInputIndex( n, plane, inputRow, inputCol ) ];
                    for( int dx = 0; dx < poolingSize; dx++ ) {
                        for( int dy = 0; dy < poolingSize; dy++ ) {
                            float thisValue = input[ getInputIndex( n, plane, inputRow + dx, inputCol + dy ) ];
                            if( thisValue > maxValue ) {
                                maxValue = thisValue;
                                selector = dx * poolingSize + dy;
                            }
                        }
                    }
                    int resultIndex = getResultIndex( n, plane, outputRow, outputCol );
                    output[ resultIndex ] = maxValue;
                    selectors[ resultIndex ] = selector;
                }
            }
        }
    }
    StatefulTimer::instance()->timeCheck("PoolingPropagateCpu::propagate end" );
//    return output;
}

