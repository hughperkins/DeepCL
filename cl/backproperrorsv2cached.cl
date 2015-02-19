// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

void copyLocal( local float *target, global float const *source, int N ) {
    int numLoops = ( N + get_local_size(0) - 1 ) / get_local_size(0);
    for( int loop = 0; loop < numLoops; loop++ ) {
        int offset = loop * get_local_size(0) + get_local_id(0);
        if( offset < N ) {
            target[offset] = source[offset];
        }
    }
}

// as calcErrorsForUpstream, but with local cache
// convolve weights with errors to produce errorsForUpstream
// workgroupid: [n][inputPlane]
// localid: [upstreamrow][upstreamcol]
// per-thread aggregation: [outPlane][filterRow][filterCol]
// need to store locally:
// - _errorBoard. size = outputBoardSizeSquared
// - _filterBoard. size = filtersizesquared
// note: currently doesnt use bias as input.  thats probably an error?
// inputs: errors :convolve: filters => errorsForUpstream
//
// global:
// errors: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4
// weights: [filterId][upstreamplane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4
// per workgroup:
// errors: [outPlane][outRow][outCol] 32 * 19 * 19 * 4 = 46KB
// weights: [filterId][filterRow][filterCol] 32 * 5 * 5 * 4 = 3.2KB
// errorsforupstream: [n][upstreamPlane][upstreamRow][upstreamCol]
void kernel calcErrorsForUpstreamCached( 
        const int batchSize,
        global const float *errorsGlobal,
        global const float *filtersGlobal, 
        global float *errorsForUpstream,
        local float *_errorBoard, 
        local float *_filterBoard ) {

    #define globalId get_global_id(0)
    #define localId get_local_id(0)
    #define workgroupId get_group_id(0)
    #define workgroupSize get_local_size(0)

    const int n = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    const int upstreamRow = localId / gInputBoardSize;
    const int upstreamCol = localId % gInputBoardSize;

    float sumWeightTimesOutError = 0;
    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {
        barrier(CLK_LOCAL_MEM_FENCE);
        copyLocal( _filterBoard, filtersGlobal + ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared, gFilterSizeSquared );
        copyLocal( _errorBoard, errorsGlobal + ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared, gOutputBoardSizeSquared );
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
            int outRow = upstreamRow + gMargin - filterRow;
            for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                int outCol = upstreamCol + gMargin - filterCol;
                if( outCol >= 0 && outCol < gOutputBoardSize && outRow >= 0 && outRow < gOutputBoardSize ) {
                    float thisWeightTimesError = 
                        _errorBoard[outRow * gOutputBoardSize + outCol] * 
                        _filterBoard[filterRow * gFilterSize + filterCol];
                    sumWeightTimesOutError += thisWeightTimesError;
                }
            }
        }
    }
    const int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
    if( localId < gInputBoardSizeSquared ) {
        errorsForUpstream[upstreamBoardGlobalOffset + localId] = sumWeightTimesOutError;
    }
}

