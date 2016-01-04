// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

static float operation(float val_one, float val_two) {
    return {{operation}};
}

kernel void per_element_op2_inplace(const int N, global float *target, global const float *source) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] = operation(target[globalId], source[globalId]);
}

kernel void per_element_op2_outofplace(const int N, global float *target, global float *one, global const float *two) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] = operation(one[globalId], two[globalId]);
}

