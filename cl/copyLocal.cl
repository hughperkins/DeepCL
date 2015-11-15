// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

static void copyLocal(local float *target, global float const *source, int N) {
    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);
    for (int loop = 0; loop < numLoops; loop++) {
        int offset = loop * get_local_size(0) + get_local_id(0);
        if (offset < N) {
            target[offset] = source[offset];
        }
    }
}

static void copyGlobal(global float *target, local float const *source, int N) {
    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);
    for (int loop = 0; loop < numLoops; loop++) {
        int offset = loop * get_local_size(0) + get_local_id(0);
        if (offset < N) {
            target[offset] = source[offset];
        }
    }
}

