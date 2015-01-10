// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

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
// some globalid structure etc
// well, except that we break into workgroups
// workgroupid: [n][upstreamPlane]
// localid: [upstreamrow][upstreamcol]
// per-thread aggregation: [outPlane][filterRow][filterCol]
// need to store locally:
// - all filters for [upstreamPlane] (or iterate). size = filtersizesquared * numfilters (but not times upstreamPlanes)
// - [upstreamPlane] board from [n] upstreamcube. size of upstreamboardsizesquared
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel calcErrorsForUpstreamCached( 
        const int batchSize,
        global const float *weights, global const float *errors, global float *errorsForUpstream,
        local float *_weightCube, local float *_upstreamBoard ) {

    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int n = workgroupId / gUpstreamNumPlanes;
    const int upstreamPlane = workgroupId % gUpstreamNumPlanes;

    const int upstreamRow = localId / gUpstreamBoardSize;
    const int upstreamCol = localId % gUpstreamBoardSize;

    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutBoardSize - 1) );
    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );
    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutBoardSize -1) );
    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );

    float sumWeightTimesOutError = 0;
    for( int outPlane = 0; outPlane < gNumOutPlanes; outPlane++ ) {
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + gMargin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = ( ( n * gNumOutPlanes 
                          + outPlane ) * gOutBoardSize
                          + outRow ) * gOutBoardSize
                          + outCol;
                float thisError = errors[resultIndex];
                int thisWeightIndex = ( ( outPlane * gUpstreamNumPlanes
                                    + upstreamPlane ) * gFilterSize
                                    + filterRow ) * gFilterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    errorsForUpstream[globalId] = sumWeightTimesOutError;
}
#endif

// how about we make each workgroup handle one upstream plane, and iterate over examples?
// for now we assume that a workgroup is large enough to have one thread per location
// but we could always simply make each thread handle two pixels I suppose :-)
// so, workgroupId is [upstreamPlane]
// localId is [upstreamRow][upstreamCol]
// we iterate over [n]
#ifdef gOutBoardSize // for previous tests that dont define it
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
    const int upstreamRow = localId / gUpstreamBoardSize;
    const int upstreamCol = localId % gUpstreamBoardSize;

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
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel convolve_errorcubes_float( 
       const int batchSize,
      global const float *errorcubes, global const float *filters, 
    global float *upstreamErrors ) {
    int globalId = get_global_id(0);

    int upstreamBoard2Id = globalId / gUpstreamBoardSizeSquared;
    int exampleId = upstreamBoard2Id / gUpstreamNumPlanes;
    int filterId = upstreamBoard2Id % gUpstreamNumPlanes;

    if( exampleId >= batchSize ) {
        return;
    }
/*
    int errorCubeOffset = exampleId * gNumOutPlanes * gOutBoardSizeSquared;
    int filterCubeOffset = filterId * gNumInputPlanes * gFilterSizeSquared;

    int localid = globalId % upstreamBoardSizeSquared;
    int upstreamRow = localid / gUpstreamBoardSize;
    int upstreamCol = localid % gUpstreamBoardSize;

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

