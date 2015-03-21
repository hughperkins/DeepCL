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

    #define globalId get_global_id(0)
    #define nPlaneCombo ( globalId / gOutputImageSizeSquared ) 
    #define outputPosCombo ( globalId % gOutputImageSizeSquared )

    const int n = nPlaneCombo / gNumPlanes;
    const int plane = nPlaneCombo % gNumPlanes;
    const int outputRow = outputPosCombo / gOutputImageSize;
    const int outputCol = outputPosCombo % gOutputImageSize;

    if( n >= batchSize ) {
        return;
    }

    int resultIndex = ( ( n
        * gNumPlanes + plane )
        * gOutputImageSize + outputRow )
        * gOutputImageSize + outputCol;
    #define error ( errors[resultIndex] )
    int selector = ( selectors[resultIndex] );
    #define drow ( selector / gPoolingSize )
    #define dcol ( selector % gPoolingSize )
    #define inputRow ( outputRow * gPoolingSize + drow )
    #define inputCol ( outputCol * gPoolingSize + dcol )
    int inputIndex = ( ( n
        * gNumPlanes + plane )
        * gInputImageSize + inputRow )
        * gInputImageSize + inputCol;
//    if( n < batchSize ) {
        errorsForUpstream[ inputIndex ] = error;
//    }
}

