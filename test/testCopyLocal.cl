// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "cl/copyBlock.cl"
#include "cl/ids.cl"
#include "cl/copyLocal.cl"

kernel void run( global const float *source, global float *destination, int N,
    local float *localBuffer ) {
    copyLocal( localBuffer, source, N );
    barrier(CLK_LOCAL_MEM_FENCE);
    copyGlobal( destination, localBuffer, N );
}

