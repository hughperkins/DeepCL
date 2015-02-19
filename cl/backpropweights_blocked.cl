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

//#define posToRow( pos ) ( ( pos >> 10 ) & (2^11-1) )
//#define posToCol( pos ) ( ( pos ) & (2^11-1) )
//#define rowColToPos( row, col ) ( ( row << 10 ) | col )
//#define linearIdToPos( linearId, base ) ( rowColToPos( ( linearId / base ), ( linearId % base )  ) )

int posToRow( int pos ) {
    return ( pos >> 10 ) & ( 2^11-1);
}
int posToCol( int pos ) {
    return ( pos ) & (2^11-1) );
}
int rowColToPos( int row, int col ) {
    return ( row << 10 ) | col );
}
int linearIdToPos( int linearId, int base ) {
    return rowColToPos( ( linearId / base ), ( linearId % base )  );
}
int posToOffset( int pos, int rowLength ) {
    return posToRow(pos) * rowLength + posToCol(pos);
}

void copyLocal( local float *target, global float const *source, int N ) {
    int numLoops = ( N + get_local_size(0) - 1 ) / get_local_size(0);
    for( int loop = 0; loop < numLoops; loop++ ) {
        int offset = loop * get_local_size(0) + get_local_id(0);
        if( offset < N ) {
            target[offset] = source[offset];
        }
    }
}

// assumes that the block will fit exactly into the target
void copyBlock( local float *target, global float const *source, 
    const int blockStart, const int blockSize, const int sourceSize ) {
    const int totalLinearSize = posToRow( blockSize ) * posToCol( blockSize );
    const int numLoops = ( totalLinearSize + get_local_size(0) - 1 ) / get_local_size(0);
    for( int loop = 0; loop < numLoops; loop++ ) {
        const int offset = get_local_id(0) + loop * get_local_size(0);
        if( offset < totalLinearSize ) {
            const int offsetAsPos = linearIdToPos( offset, posToRow( blockSize ) );
            target[ offset ] = source[ posToOffset( blockStart + offsetAsPos ) ];  
        }
    }
}

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
        barrier(CLK_LOCAL_MEM_FENCE);
        copyLocal( _imageBoard, images + ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared, 
            gInputBoardSizeSquared );
        copyLocal( _errorBoard, errors + ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared,
            gOutputBoardSizeSquared );
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

