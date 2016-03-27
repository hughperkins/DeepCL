// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

kernel void cl_memset(global float *target, const float value, const int N) {
    #define globalId get_global_id(0)
    if ((int)globalId < N) {
        target[globalId] = value;
    }
}

