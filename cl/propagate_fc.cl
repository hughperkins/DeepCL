// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) ( 1.7159f * tanh( 0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif

#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroupid 
// localid: 
//  each thread is assigned to: one row, of one output plane
//  workgroup is assigned to: same row, from each output plane
//  each thread iterates over: [n]
// local memory: one row from each output, = 128 * 19 * 4 = 9.8KB
//             1 * input row = "0.076KB"
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputboardsize (mandatory)
//   inputboardsize == 19
//   filtersize == 19
//   outputBoardSize == 1
//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32-128
//   inputboardsize around 19, not too small
#if gFilterSize == gInputBoardSize && gPadZeros == 0
void kernel propagate_fc( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterBoard ) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    float sum = 0;
    for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
        int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
            }
        }
        const int filterGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;
        for( int i = 0; i < numFilterPixelsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gFilterSizeSquared ) {
                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gOutBoardSizeSquared ) {
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                int inputboardrowoffset = inputRow * gUpstreamBoardSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // results are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;
    if( localId < gOutBoardSizeSquared ) {
        results[resultIndex ] = ACTIVATION_FUNCTION(sum);
//        results[resultIndex ] = 123;
    }
}
#endif
#endif
#endif

#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroupid [n][outputplane]
// localid: [filterrow][filtercol]
//  each thread iterates over: [inplane]
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputboardsize (mandatory)
//   outputBoardSize == 1
//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32
//   inputboardsize around 19, not too small
#if gFilterSize == gInputBoardSize && gPadZeros == 0
void kernel propagate_filter_matches_inboard( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterBoard ) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    float sum = 0;
    for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
        int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
            }
        }
        const int filterGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;
        for( int i = 0; i < numFilterPixelsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gFilterSizeSquared ) {
                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gOutBoardSizeSquared ) {
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                int inputboardrowoffset = inputRow * gUpstreamBoardSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // results are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;
    if( localId < gOutBoardSizeSquared ) {
        results[resultIndex ] = ACTIVATION_FUNCTION(sum);
//        results[resultIndex ] = 123;
    }
}
#endif
#endif
#endif


