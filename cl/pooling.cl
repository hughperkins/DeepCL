// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// every plane is independent
// every example is independent
// so, globalid can be: [n][plane][outputRow][outputCol]
kernel void propagateNaive( const int batchSize, global const float *input, global int *selectors, global float *output ) {
    const int globalId = get_global_id(0);

    const int intraBoardOffset = globalId % gOutputBoardSizeSquared;
    const int outputRow = intraBoardOffset / gOutputBoardSize;
    const int outputCol = intraBoardOffset % gOutputBoardSize;

    const int board2dIdx = globalId / gOutputBoardSizeSquared;
    const int plane = board2dIdx % gNumPlanes;
    const int n = board2dIdx / gNumPlanes;

    if( n >= batchSize ) {
        return;
    }

    const int inputRow = outputRow * gPoolingSize;
    const int inputCol = outputCol * gPoolingSize;
    const int inputBoardOffset = ( n * gNumPlanes + plane ) * gInputBoardSizeSquared;
    int selector = 0;
    int poolInputOffset = inputBoardOffset + inputRow * gInputBoardSize + inputCol;
    float maxValue = input[ poolInputOffset ];
    for( int dRow = 0; dRow < gPoolingSize; dRow++ ) {
        for( int dCol = 0; dCol < gPoolingSize; dCol++ ) {
            bool process = ( inputRow + dRow < gInputBoardSize ) && ( inputCol + dCol < gInputBoardSize );
            if( process ) {
                float thisValue = input[ poolInputOffset + dRow * gInputBoardSize + dCol ];
                if( thisValue > maxValue ) {
                    maxValue = thisValue;
                    selector = dRow * gPoolingSize + dCol;
                }
            }
        }
    }
    output[ globalId ] = maxValue;
    selectors[ globalId ] = selector;
}

