// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// globalId: [outPlane][inputPlane][filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
void kernel backprop_floats( const float learningRateMultiplier,
        const int batchSize, 
         global const float *errors, global const float *images, 
        global float *weights
        #ifdef BIASED
            , global float *biasWeights
        #endif
 ) {
    int globalId = get_global_id(0);
    if( globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize ) {
        return;
    }

    int IntraFilterOffset = globalId % gFilterSizeSquared;
    int filterRow = IntraFilterOffset / gFilterSize;
    int filterCol = IntraFilterOffset % gFilterSize;

    int filter2Id = globalId / gFilterSizeSquared;
    int outPlane = filter2Id / gInputPlanes;
    int upstreamPlane = filter2Id % gInputPlanes;

    float thiswchange = 0;
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
        for( int outRow = 0; outRow < gOutputBoardSize; outRow++ ) {
            int upstreamRow = outRow - gMargin + filterRow;
            for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputBoardSize
                    && upstreamCol < gInputBoardSize;
                if( proceed ) {
                    int resultIndex = ( ( n * gNumFilters 
                              + outPlane ) * gOutputBoardSize
                              + outRow ) * gOutputBoardSize
                              + outCol;
                    float error = errors[resultIndex];
                    int upstreamDataIndex = ( ( n * gInputPlanes 
                                     + upstreamPlane ) * gInputBoardSize
                                     + upstreamRow ) * gInputBoardSize
                                     + upstreamCol;
                    float upstreamResult = images[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * error;
                    thiswchange += thisimagethiswchange;
    #ifdef BIASED
                    thisbiaschange += error;
    #endif
                }
            }
        }
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    weights[ globalId ] += - learningRateMultiplier * thiswchange;
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && IntraFilterOffset == 0;
    if( writeBias ) {
        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;
    }
#endif
}



