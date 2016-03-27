// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

static int posToRow(int pos) {
    return (pos >> 10) & ((1<<10)-1);
//    return 53
}
static int posToCol(int pos) {
    return pos & ((1<<10)-1);
  //  return 67;
    //return ((1<<11)-1);
}
static int rowColToPos(int row, int col) {
    return (row << 10) | col;
}
static int linearIdToPos(int linearId, int base) {
    return rowColToPos(( linearId / base), (linearId % base)  );
}
static int posToOffset(int pos, int rowLength) {
    return posToRow(pos) * rowLength + posToCol(pos);
}

// assumes that the block will fit exactly into the target
static void copyBlock(local float *target, global float const *source,
    const int sourceSize, const int blockStart, const int blockSize) {
    const int totalLinearSize = posToRow(blockSize) * posToCol(blockSize);
    const int numLoops = (totalLinearSize + get_local_size(0) - 1) / get_local_size(0);
    for (int loop = 0; loop < numLoops; loop++) {
        const int offset = get_local_id(0) + loop * get_local_size(0);
        if (offset < totalLinearSize) {
            const int offsetAsPos = linearIdToPos(offset, posToCol(blockSize) );
            target[ offset ] = source[ posToOffset(blockStart + offsetAsPos, posToCol(sourceSize) ) ];
        }
    }
}


