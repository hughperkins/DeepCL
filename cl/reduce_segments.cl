// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

kernel void reduce_segments(const int numSegments, const int segmentLength, 
        global float const *in, global float* out) {
    const int globalId = get_global_id(0);
    const int segmentId = globalId;

    if (segmentId >= numSegments) {
        return;
    }

    float sum = 0;
    global const float *segment = in + segmentId * segmentLength;
    for (int i = 0; i < segmentLength; i++) {
        sum += segment[i];
    }
    out[segmentId] = sum;
}


