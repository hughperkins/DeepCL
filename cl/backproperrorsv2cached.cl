// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

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
// per workgroup:
// errors: [outPlane][outRow][outCol] 32 * 19 * 19 * 4 = 46KB
// weights: [filterId][filterRow][filterCol] 32 * 5 * 5 * 4 = 3.2KB
void kernel calcErrorsForUpstreamCached( 
        const int batchSize,
        global const float *errorsGlobal,
        global const float *filtersGlobal, 
        global float *errorsForUpstream,
        local float *_errorBoard, 
        local float *_filterBoard ) {

    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int n = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    const int upstreamRow = localId / gInputBoardSize;
    const int upstreamCol = localId % gInputBoardSize;

    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutputBoardSize - 1) );
    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );
    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutputBoardSize -1) );
    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );

    const int filterPixelCopiesPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;
    const int errorPixelCopiesPerThread = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
    const int pixelCopiesPerThread = max( filterPixelCopiesPerThread, errorPixelCopiesPerThread );

    float sumWeightTimesOutError = 0;
    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {
        const int filterBoardGlobalOffset =( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared;
        const int errorBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < pixelCopiesPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gFilterSizeSquared ) {
                _filterBoard[ thisOffset ] = filtersGlobal[ filterBoardGlobalOffset + thisOffset ];
            }
            if( thisOffset < gOutputBoardSizeSquared ) {
                _errorBoard[ thisOffset ] = errorsGlobal[ errorBoardGlobalOffset + thisOffset ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
//        if( globalId == 0 ) {
//            for( int i = 0; i < gFilterSizeSquared; i++ ) {
//                errorsForUpstream[ (outPlane+1)*100 + i ] = _filterBoard[i];
//            }
//        }
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + gMargin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = outRow * gOutputBoardSize + outCol;
                float thisError = _errorBoard[resultIndex];
                int thisWeightIndex = filterRow * gFilterSize + filterCol;
                float thisWeight = _filterBoard[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    const int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
    if( localId < gInputBoardSizeSquared ) {
        errorsForUpstream[upstreamBoardGlobalOffset + localId] = sumWeightTimesOutError;
    }
}

