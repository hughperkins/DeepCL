// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// one over the value

kernel void array_inv(
        const int N,
        global float *data) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    data[globalId] = 1.0f / data[globalId];
}

