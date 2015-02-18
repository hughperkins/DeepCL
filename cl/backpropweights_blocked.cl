// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// blockRow, blockCol
// blockPos
// margin
// localInRow, localInCol
// localOutRow, localOutCol
// 

//typedef struct tag_block {
//    int pos;
//} block;

#define posToRow( pos ) ( ( pos >> 10 ) & (2^11-1) )
#define posToCol( pos ) ( ( pos ) & (2^11-1) )
#define rowColToPos( row, col ) ( ( row << 10 ) | col )
#define linearIdToPos( linearId, base ) ( rowColToPos( ( linearId / base ), ( linearId % base )  ) )

// workgroupId: [outputPlane][inputPlane][blockRow][blockCol]
// localId: [filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
// local: errorboard: blockSize * blockSize
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
    #define globalId ( get_global_id(0) )
    #define localId ( get_local_id(0)  )
    #define workgroupId ( get_group_id(0) )
    #define workgroupSize ( get_local_size(0) )

//    const int filterRow = localId / gFilterSize;
//    const int filterCol = localId % gFilterSize;
    const int filterPos = linearIdToPos( localId, gFilterSize )
    const int inOutPlane = linearIdToPos( workgroupId, gInputPlanes )

//    #define outPlane ( workgroupId / gInputPlanes )
//    #define upstreamPlane ( workgroupId % gInputPlanes )

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
	    { // these parantheses are to prevent leakage of names to outside of this block
           // I'm hoping it reduces register pressure, but probably somewhat compiler-dependent
            int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
            // need to fetch the board, but it's bigger than us, so will need to loop...
            const int numLoopsForUpstream = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numLoopsForUpstream; i++ ) {
                const int imageOffset = i * workgroupSize + localId;
                if( imageOffset < gInputBoardSizeSquared ) {
                    _imageBoard[imageOffset] = images[ upstreamBoardGlobalOffset + imageOffset ];
                }
            }
	    }
	    {
            int resultBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;
            int numLoopsForResults = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
            for( int i = 0; i < numLoopsForResults; i++ ) {
                int thisOffset = i * workgroupSize + localId;
                if( thisOffset < gOutputBoardSizeSquared ) {
                    _errorBoard[thisOffset ] = errors[resultBoardGlobalOffset + thisOffset];
                }
            }
	    }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gFilterSizeSquared ) {
            for( int outRow = 0; outRow < gOutputBoardSize; outRow++ ) {
                int upstreamRow = outRow - gMargin + filterRow;
                for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                    const int upstreamCol = outCol - gMargin + filterCol;
                    #define proceed ( upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputBoardSize && upstreamCol < gInputBoardSize )
                    if( proceed ) {
                        // these defines reduce register pressure, compared to const
                        // giving a 40% speedup on nvidia :-)
                        #define resultIndex ( outRow * gOutputBoardSize + outCol )
                        #define error ( _errorBoard[resultIndex] )
                        //const float error = _errorBoard[resultIndex];
                        #define upstreamDataIndex ( upstreamRow * gInputBoardSize + upstreamCol )
                        #define upstreamResult ( _imageBoard[upstreamDataIndex] )
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
        weights[ workgroupId * gFilterSizeSquared + localId ] = learningRateMultiplier * thiswchange;
    }
#ifdef BIASED
    #define writeBias ( upstreamPlane == 0 && localId == 0 )
    if( writeBias ) {
        biasWeights[outPlane] = learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

