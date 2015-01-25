// Copyright Hugh Perkins 2014 hughperkins at gmail
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

// workgroupId: [outputPlane][inputPlane]
// localId: [filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
// local: errorboard: outputBoardSize * outputBoardSize
//        imageboard: inputBoardSize * inputBoardSize
void kernel backprop_floats_withscratch_dobias( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *errors, global const float *images, 
        global float *weights,
        #ifdef BIASED
             global float *biasWeights,
        #endif
        local float *_errorBoard, local float *_imageBoard
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
        int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
        // need to fetch the board, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gInputBoardSizeSquared ) {
                _imageBoard[thisOffset] = images[ upstreamBoardGlobalOffset + thisOffset ];
            }
        }
        int resultBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;
        int numLoopsForResults = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutputBoardSizeSquared ) {
                _errorBoard[thisOffset ] = errors[resultBoardGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gFilterSizeSquared ) {
            for( int outRow = 0; outRow < gOutputBoardSize; outRow++ ) {
                int upstreamRow = outRow - gMargin + filterRow;
                for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                    int upstreamCol = outCol - gMargin + filterCol;
                    bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputBoardSize
                        && upstreamCol < gInputBoardSize;
                    if( proceed ) {
                        int resultIndex = outRow * gOutputBoardSize + outCol;
                        float error = _errorBoard[resultIndex];
                        int upstreamDataIndex = upstreamRow * gInputBoardSize + upstreamCol;
                        float upstreamResult = _imageBoard[upstreamDataIndex];
                        thiswchange += upstreamResult * error;
    #ifdef BIASED
                        thisbiaschange += error;
    #endif
                    }
                }
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weights[ workgroupId * gFilterSizeSquared + localId ] += - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && localId == 0;
    if( writeBias ) {
        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

// workgroupId: [outputPlane][inputPlane]
// localId: [filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
// local: errorboard: outputBoardSize * outputBoardSize
//        imageboard: inputBoardSize * inputBoardSize
// specific characteristic: load one stripe of each image at a time,
// so we dont run out of memory
// number of stripes set in: gNumStripes
// note that whilst we can stripe the errors simply, 
// we actually need to add a half-filter widthed additional few rows
// onto the images stripe, otherwise we will be missing data
//   we will call the size of the non-overlapping image stripes: gInputStripeInnerSize
//      the outersize, including the two margins is: gInputStripeOuterSize
//      of course, the first and last stripes will be missing a bit off the top/bottom, where the 
//      corresponding outer margin would be
void kernel backprop_floats_withscratch_dobias_striped( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *errors, global const float *images, 
        global float *weights,
        #ifdef BIASED
             global float *biasWeights,
        #endif
        local float *_errorStripe, local float *_imageStripe
 ) {
    // gHalfFilterSize
    // gInputBoardSize
    //
    // gInputStripeMarginRows => basically equal to gHalfFilterSize
    // gInputStripeInnerNumRows = gInputBoardSize / gNumStripes
    // gInputStripeOuterNumRows = gInputStripeInnerNumRows + 2 * gHalfFilterSize  (note: one row less than
    //                                                         if we just added gFilterSize)
    // gInputStripeInnerSize = gInputStripeInnerNumRows * gInputBoardSize
    // gInputStripeOuterSize = gInputStripeOuterNumRows * gInputBoardSize
    // gInputStripeMarginSize = gInputStripeMarginRows * gInputBoardSize
    //
    // gOutputStripeNumRows
    // gOutputStripeSize

    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    const int numLoopsForImageStripe = ( gInputStripeOuterSize + workgroupSize - 1 ) / workgroupSize;
    const int numLoopsForErrorStripe = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
    for( int n = 0; n < batchSize; n++ ) {
        const int imageBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
        const int imageBoardGlobalOffsetAfter = imageBoardGlobalOffset + gInputBoardSizeSquared;
        const int errorBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;
        for( int stripe = 0; stripe < gNumStripes; stripe++ ) {
            int imageStripeOffset = imageBoardGlobalOffset + stripe * gInputStripeInnerSize
                               - gInputStripeMarginSize;
            // need to fetch the board, but it's bigger than us, so will need to loop...
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numLoopsForImageStripe; i++ ) {
                int thisLocalOffset = i * workgroupSize + localId;
                int thisGlobalOffset = imageStripeOffset + thisOffset;
                bool process = thisOffset < gInputStripeOuterSize 
                    && thisGlobalOffset >= imageBoardGlobalOffset && thisGlobalOffset < imageBoardGlobalOffsetAfter;
                if( process ) {
                    _imageStripe[thisLocalOffset] = images[ thisGlobalOffset ];
                }
            }
            int errorStripeOffset = errorBoardGlobalOffset + stripe * gOutputStripeSize;
            for( int i = 0; i < numLoopsForErrorStripe; i++ ) {
                int thisOffset = i * workgroupSize + localId;
                if( thisOffset < gOutputStripeSize ) {
                    _errorStripe[thisOffset ] = errors[errorStripeOffset + thisOffset];
                }
            }
            const int stripeOutRowStart = stripe * gOutputStripeNumRows;
            const int stripeOutRowEndExcl = stripeOutRowStart + gOutputStripeNumRows;
            barrier(CLK_LOCAL_MEM_FENCE);
            if( localId < gFilterSizeSquared ) {
                for( int outRow = stripeOutRowStart; outRow < stripeOutRowEndExcl; outRow++ ) {
                    int upstreamRow = outRow - gMargin + filterRow;
                    for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                        int upstreamCol = outCol - gMargin + filterCol;
                        bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputBoardSize
                            && upstreamCol < gInputBoardSize; // dont need to check more than this, since
                                                              // the inputstripe margin means we have
                        if( proceed ) {
                            int resultIndex = outRow * gOutputBoardSize + outCol;
                            float error = _errorStripe[resultIndex - stripe * gOutputStripeSize];
                            int upstreamDataIndex = upstreamRow * gInputBoardSize + upstreamCol;
                            // next line segfaults on nvidia, out of bounds
                            float upstreamResult = _imageStripe[upstreamDataIndex +  gInputStripeMarginSize
                                        - stripe * gInputStripeInnerSize ];
                            thiswchange += upstreamResult * error;
        #ifdef BIASED
                            thisbiaschange += error;
        #endif
                        }
                    }
                }
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weights[ workgroupId * gFilterSizeSquared + localId ] += - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && localId == 0;
    if( writeBias ) {
        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

