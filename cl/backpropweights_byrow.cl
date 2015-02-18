// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// reminder:
// - for backprop weights, we take one plane from one image, convolve with one plane from the output
//   and reduce over n

// concept:
// - here, we process only single row from the input/output cube (same row from each)
//   and then we will need to reduce the resulting weight changes over the rows, in a separate kernel
// - this assumes that the filter cubes are small, so reducing over 32 or so of them is not a big task

// here, we will use one workgroup for one row of a single pair of input/output planes
// and sum over n
// workgroup: [outputPlane][inputPlane][inputRow]
// localid: [filterRow][filterCol]
// weightChanges1: [outputPlane][inputPlane][filterRow][filterCol][inputRow]
kernel void backprop_weights( const int batchSize,
     global float const *errors, global float const *input, global float *weightChanges1 ) {
    #define globalId ( get_global_id(0) )
    #define workgroupId ( get_group_id(0) )
    #define localId ( get_local_id(0) )
    
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;
    const int inputRow = workgroupId % gInputBoardSize;
    #define outInCombo = ( workgroupId / gInputBoardSize )
    const int outputPlane = outInCombo / gNumInputPlanes;
    const int inputPlane = outInCombo % gNumInputPlanes;

    for( int n = 0; n < batchSize; n++ ) {
        
    }
}

