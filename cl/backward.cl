// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
//  - none

// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]
// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB
// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB
// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB
void kernel calcGradInput( 
        const int batchSize,
        global const float *gradOutput, global float *weights, global float *gradInput) {
    int globalId = get_global_id(0);

    const int upstreamImage2dId = globalId / gInputSizeSquared;

    const int intraImageOffset = globalId % gInputSizeSquared;
    const int upstreamRow = intraImageOffset / gInputSize;
    const int upstreamCol = intraImageOffset % gInputSize;

    const int upstreamPlane = upstreamImage2dId % gInputPlanes;
    const int n = upstreamImage2dId / gInputPlanes;

    if (n >= batchSize) {
        return;
    }

    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));
    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);
    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));
    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);

    float sumWeightTimesOutError = 0;
    // aggregate over [outPlane][outRow][outCol]
    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {
        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
            int outRow = upstreamRow + gMargin - filterRow;
            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = (( n * gNumFilters 
                          + outPlane) * gOutputSize
                          + outRow) * gOutputSize
                          + outCol;
                float thisError = gradOutput[resultIndex];
                int thisWeightIndex = (( outPlane * gInputPlanes
                                    + upstreamPlane) * gFilterSize
                                    + filterRow) * gFilterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    gradInput[globalId] = sumWeightTimesOutError;
}

