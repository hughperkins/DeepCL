// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
//  - none

// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]
// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB
// errors: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB
// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB
void kernel calcErrorsForUpstream( 
        const int batchSize,
        global const float *errors, global float *weights, global float *errorsForUpstream ) {
    int globalId = get_global_id(0);

    const int upstreamImage2dId = globalId / gInputImageSizeSquared;

    const int intraImageOffset = globalId % gInputImageSizeSquared;
    const int upstreamRow = intraImageOffset / gInputImageSize;
    const int upstreamCol = intraImageOffset % gInputImageSize;

    const int upstreamPlane = upstreamImage2dId % gInputPlanes;
    const int n = upstreamImage2dId / gInputPlanes;

    if( n >= batchSize ) {
        return;
    }

    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutputImageSize - 1) );
    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );
    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutputImageSize -1) );
    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );

    float sumWeightTimesOutError = 0;
//    int inputDataIndex = globalId;
//    float inputDataValue = inputData[inputDataIndex];
//    float inputDeriv = ACTIVATION_DERIV( inputDataValue );
    // aggregate over [outPlane][outRow][outCol]
    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + gMargin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = ( ( n * gNumFilters 
                          + outPlane ) * gOutputImageSize
                          + outRow ) * gOutputImageSize
                          + outCol;
                float thisError = errors[resultIndex];
                int thisWeightIndex = ( ( outPlane * gInputPlanes
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
    //errorsForUpstream[globalId] = sumWeightTimesOutError * inputDeriv;
}

