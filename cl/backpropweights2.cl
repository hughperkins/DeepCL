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


