// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// inplane and outplane are always identical, 1:1 mapping, so can just write `plane`
// errors: [n][plane][outrow][outcol]
// selectors: [n][plane][outrow][outcol]
// errorsForUpstream: [n][plane][inrow][incol]
// wont use workgroups (since 'naive')
// one thread per: [n][plane][outrow][outcol]
// globalId: [n][plane][outrow][outcol]
kernel void backprop_errors( const int batchSize, 
    global const float *errors, global const int *selectors, global float *errorsForUpstream ) {
//    const int globalId = get_global_id(0);


    memset( errorsForUpstream, 0, sizeof( float ) * getInputSize( batchSize ) );
    for( int n = 0; n < batchSize; n++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outputRow = 0; outputRow < outputBoardSize; outputRow++ ) {
                int inputRow = outputRow * poolingSize;
                for( int outputCol = 0; outputCol < outputBoardSize; outputCol++ ) {
                    int inputCol = outputCol * poolingSize;
                    int resultIndex = getResultIndex( n, plane, outputRow, outputCol );
                    float error = errors[resultIndex];
                    int selector = selectors[resultIndex];
                    int drow = selector / poolingSize;
                    int dcol = selector % poolingSize;
                    int inputIndex = getInputIndex( n, plane, inputRow + drow, inputCol + dcol );
                    errorsForUpstream[ inputIndex ] = error;
                }
            }
        }
    }
}

