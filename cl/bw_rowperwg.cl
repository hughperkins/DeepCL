// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// workgroupId: [outputPlane][inputPlane][inputRow]
// localId: [filterRow][filterCol]
// per-thread iteration: [n][outputCol]
// local: errorboard: outputBoardSize
//        imageboard: inputBoardSize
// output weight changes: [outputPlane][inputPlane][filterRow][filterCol][outRow]
void kernel backprop_weights( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *errors, global const float *images, 
        global float *weightChanges,
        #ifdef BIASED
             global float *biasWeightChanges,
        #endif
        local float *_errorBoard, local float *_imageBoard
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int inputRow = workgroupId % gInputBoardSize;
    const int outputPlane = ( workgroupId / gInputBoardSize ) / gInputPlanes;
    const int inputPlane = ( workgroupId / gInputBoardSize ) % gInputPlanes;

    // weightchanges:     [outputPlane][inputPlane][filterRow][filterCol][outRow]
    //       aggregate over:  [outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
        int upstreamBoardGlobalOffset = ( n * gInputPlanes + inputPlane ) * gInputBoardSizeSquared;
        // need to fetch the board, but it's bigger than us, so will need to loop...
        const int numLoopsForUpstream = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gInputBoardSizeSquared ) {
                _imageBoard[thisOffset] = images[ upstreamBoardGlobalOffset + thisOffset ];
            }
        }
        int resultBoardGlobalOffset = ( n * gNumFilters + outputPlane ) * gOutputBoardSizeSquared;
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
                int inputRow = outRow - gMargin + filterRow;
                for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                    int inputCol = outCol - gMargin + filterCol;
                    bool proceed = inputRow >= 0 && inputCol >= 0 && inputRow < gInputBoardSize
                        && inputCol < gInputBoardSize;
                    if( proceed ) {
                        int resultIndex = outRow * gOutputBoardSize + outCol;
                        float error = _errorBoard[resultIndex];
                        int upstreamDataIndex = inputRow * gInputBoardSize + inputCol;
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
        weights[ workgroupId * gFilterSizeSquared + localId ] -= learningRateMultiplier * thiswchange;
    }
#ifdef BIASED
    bool writeBias = inputPlane == 0 && localId == 0;
    if( writeBias ) {
        biasWeights[outputPlane] -= learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outputPlane][inputPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

