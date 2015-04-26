// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

kernel void propagateNaive(
        const int N, 
        global const unsigned char *mask,
        global const float *input,
        global float *output ) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    output[globalId] = mask[globalId] == 1 ? input[globalId] : 0.0f;
}

kernel void backpropNaive(
        const int N,
        global const unsigned char *mask,
        global const float *errors,
        global float *output) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    output[globalId] = mask[globalId] == 1 ? errors[globalId] : 0.0f;
}

