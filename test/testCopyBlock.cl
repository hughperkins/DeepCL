// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "cl/copyBlock.cl"
#include "cl/ids.cl"
#include "cl/copyLocal.cl"

kernel void testPos( global const float *in, global float *out ) {
    if( get_global_id(0) == 0 ) {
        out[0] = posToRow( in[0] );
        out[1] = posToCol( in[0] );
        int pos = rowColToPos( in[1], in[2] );
        out[2] = pos;
        out[3] = posToRow(pos);
        out[4] = posToCol(pos);
    }
}

kernel void run( global const float *source, global float *destination, int sourceSize, int blockPos, int blockSize,
    local float *localBuffer ) {
    copyBlock( localBuffer, source, sourceSize, blockPos, blockSize );
    //copyLocal( localBuffer, source, posToRow( blockSize ) * posToCol( blockSize ) );
    barrier(CLK_LOCAL_MEM_FENCE);
    copyGlobal( destination, localBuffer, posToRow( blockSize ) * posToCol( blockSize ) );
}

