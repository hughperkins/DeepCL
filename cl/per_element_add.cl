// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

kernel void per_element_add(const int N, global float *target, global const float *source) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] += source[globalId];
}

// adds source to target
// tiles source as necessary, according to tilingSize
kernel void per_element_tiled_add(const int N, const int tilingSize, global float *target, global const float *source) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] += source[globalId % tilingSize];
}

kernel void repeated_add(const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] += source[ (globalId / repeatSize) % sourceSize ];
}

