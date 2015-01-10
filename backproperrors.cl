// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

#define getFilterBoardOffset( filter, inputPlane ) ( filter * gInputPlanes + inputPlane ) * gFilterSizeSquared
#define getResultBoardOffset( n, filter ) ( n * gNumFilters + filter ) * gOutputBoardSizeSquared

// handle lower layer...
// errors for upstream look like [n][inPlane][inRow][inCol]
// need to aggregate over: [outPlane][outRow][outCol] (?)
// need to backprop errors along each possible weight
// each upstream feeds to:
//    - each of our filters (so numPlanes filters)
//    - each of our outpoint points (so boardSize * boardSize)
// errors are provider per [n][inPlane][inRow][inCol]
// globalid is structured as: [n][upstreamPlane][upstreamRow][upstreamCol]
// there will be approx 128 * 32 * 28 * 28 = 3 million threads :-P
// grouped into 4608 workgroups
// maybe we want fewer than this?
// note: currently doesnt use bias as input.  thats probably an error?
void kernel calcErrorsForUpstream( 
        const int upstreamNumPlanes, const int upstreamBoardSize, const int filterSize, 
        const int outNumPlanes, const int outBoardSize,
        const int padZeros,
        global const float *weights, global const float *errors, global float *errorsForUpstream ) {
    int globalId = get_global_id(0);
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;

    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int upstreamBoard2dId = globalId / upstreamBoardSizeSquared;

    const int intraBoardOffset = globalId % upstreamBoardSizeSquared;
    const int upstreamRow = intraBoardOffset / upstreamBoardSize;
    const int upstreamCol = intraBoardOffset % upstreamBoardSize;

    const int upstreamPlane = upstreamBoard2dId % upstreamNumPlanes;
    const int n = upstreamBoard2dId / upstreamNumPlanes;

    const int minFilterRow = max( 0, upstreamRow + margin - (outBoardSize - 1) );
    const int maxFilterRow = min( filterSize - 1, upstreamRow + margin );
    const int minFilterCol = max( 0, upstreamCol + margin - (outBoardSize -1) );
    const int maxFilterCol = min( filterSize - 1, upstreamCol + margin );

    float sumWeightTimesOutError = 0;
    // aggregate over [outPlane][outRow][outCol]
    for( int outPlane = 0; outPlane < outNumPlanes; outPlane++ ) {
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + margin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + margin - filterCol;
                int resultIndex = ( ( n * outNumPlanes 
                          + outPlane ) * outBoardSize
                          + outRow ) * outBoardSize
                          + outCol;
                float thisError = errors[resultIndex];
                int thisWeightIndex = ( ( outPlane * upstreamNumPlanes
                                    + upstreamPlane ) * filterSize
                                    + filterRow ) * filterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    errorsForUpstream[globalId] = sumWeightTimesOutError;
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
#ifdef gOutputBoardSize // for previous tests that dont define it
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
        const int filterBoardGlobalOffset =( outPlane * gNumFilters + upstreamPlane ) * gFilterSizeSquared;
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
#endif

// how about we make each workgroup handle one upstream plane, and iterate over examples?
// for now we assume that a workgroup is large enough to have one thread per location
// but we could always simply make each thread handle two pixels I suppose :-)
// so, workgroupId is [upstreamPlane]
// localId is [upstreamRow][upstreamCol]
// we iterate over [n]
#ifdef gOutputBoardSize // for previous tests that dont define it
/*
void kernel calcErrorsForUpstream2( 
        const int batchSize,
        global const float *weightsGlobal, global const float *errorsGlobal, 
        global float *errorsForUpstreamGlobal,
        local float *_weightBoard, local float *_errorBoard ) {
    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);

    const int upstreamPlane = workgroupId;
    const int upstreamRow = localId / gInputBoardSize;
    const int upstreamCol = localId % gInputBoardSize;

    const int 
    if( localId < filterSizeSquared ) {
        _weightBoard[localId] = weightsGlobal[localId];
    }

    for( int n = 0; n < batchSize; n++ ) {
        float sumWeightTimesOutError = 0;
        // aggregate over [outPlane][outRow][outCol]
        for( int outPlane = 0; outPlane < outNumPlanes; outPlane++ ) {
            for( int outRow = 0; outRow < outBoardSize; outRow++ ) {
                // need to derive filterRow and filterCol, given outRow and outCol
                int filterRow = upstreamRow + margin - outRow;
                for( int outCol = 0; outCol < outBoardSize; outCol++ ) {
                   // need to derive filterRow and filterCol, given outRow and outCol
                    int filterCol = upstreamCol + margin - outCol;
                    int resultIndex = ( ( n * outNumPlanes 
                              + outPlane ) * outBoardSize
                              + outRow ) * outBoardSize
                              + outCol;
                    float thisError = errors[resultIndex];
                    int thisWeightIndex = ( ( outPlane * upstreamNumPlanes
                                        + upstreamPlane ) * filterSize
                                        + filterRow ) * filterSize
                                        + filterCol;
                    float thisWeight = weights[thisWeightIndex];
                    float thisWeightTimesError = thisWeight * thisError;
                    sumWeightTimesOutError += thisWeightTimesError;
                }
            }
        }
        errorsForUpstream[globalId] = sumWeightTimesOutError;
    }
}
*/
#endif

// so, we're just going to convolve the errorcubes with our filter cubes...
// like propagate, but easier, since no activation function, and no biases
// errorcubes (*) filters => errors
// for propagation we had:
//   images are organized like [imageId][plane][row][col]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//   results are organized like [imageid][filterid][row][col]
//   global id is organized like results, ie: [imageid][filterid][row][col]
//   - no local memory used currently
//   - each thread:
//     - loads a whole board
//     - loads a whole filter
//     - writes one output
// we will have the other way around:
//   errorcubes are organized like [imageid][outPlane][outRow][outCol]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//        (so we will swap filterid and inplane around when referencing filters, kindof)
//  globalid will be organized like upstreamresults, ie [imageid][upstreamplane][upstreamrow][upstreamcol]
#ifdef gOutputBoardSize // for previous tests that dont define it
void kernel convolve_errorcubes_float( 
       const int batchSize,
      global const float *errorcubes, global const float *filters, 
    global float *upstreamErrors ) {
    int globalId = get_global_id(0);

    int upstreamBoard2Id = globalId / gInputBoardSizeSquared;
    int exampleId = upstreamBoard2Id / gInputPlanes;
    int filterId = upstreamBoard2Id % gInputPlanes;

    if( exampleId >= batchSize ) {
        return;
    }
/*
    int errorCubeOffset = exampleId * gOutPlanes * gOutputBoardSizeSquared;
    int filterCubeOffset = filterId * gNumInputPlanes * gFilterSizeSquared;

    int localid = globalId % upstreamBoardSizeSquared;
    int upstreamRow = localid / gInputBoardSize;
    int upstreamCol = localid % gInputBoardSize;

    float sum = 0;
// ====in progress
    int minm = padZeros ? max( -halfFilterSize, -outputRow ) : -halfFilterSize;
// ====to do
    int maxm = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputRow ) : halfFilterSize;
    int minn = padZeros ? max( -halfFilterSize, -outputCol ) : - halfFilterSize;
    int maxn = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputCol ) : halfFilterSize;
    int inputPlane = 0;
    while( inputPlane < numInputPlanes ) {
        int inputBoardOffset = inputCubeOffset + inputPlane * inputBoardSizeSquared;
        int filterBoardOffset = filterCubeOffset + inputPlane * filterSizeSquared;
        int m = minm;
        while( m <= maxm ) {
            int inputRow = outputRow + m + ( padZeros ? 0 : halfFilterSize );
            int inputboardrowoffset = inputBoardOffset + inputRow * inputBoardSize;
            int filterrowoffset = filterBoardOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while( n <= maxn ) {
                int inputCol = outputCol + n + ( padZeros ? 0 : halfFilterSize );
                sum += images[ inputboardrowoffset + inputCol] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    results[globalId] = sum;*/
}
#endif

