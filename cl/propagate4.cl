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

#ifdef gOutputBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroup id organized like: [imageid][outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// results are organized like [imageid][filterid][row][col]
void kernel propagate_4_by_n_outplane_smallercache( const int batchSize,
    const int pixelsPerThread,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterCube, local float *_pixelSums ) {
    const int globalId = get_global_id(0);

    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int numUpstreamsPerThread = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
    const int numFilterPixelsPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;

    local float *_myPixelSums = _pixelSums + pixelsPerThread * localId;

    for( int pixel = 0; pixel < pixelsPerThread; pixel++ ) {
        _myPixelSums[pixel] = 0.0f;
    }


    const int outputRow = localId / gOutputBoardSize;
    const int outputCol = localId % gOutputBoardSize;

    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;
    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutputBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;

    for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {
        int thisUpstreamBoardOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gInputBoardSizeSquared ) {
                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
            }
        }
        const int filterGlobalOffset = ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared;
        for( int i = 0; i < numFilterPixelsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gFilterSizeSquared ) {
                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for( int pixel = 0; pixel < pixelsPerThread; pixel++ ) {
            const int virtualLocalId = localId + pixel * workgroupSize;

            float thissum = 0;
            if( virtualLocalId < gOutputBoardSizeSquared ) {
                for( int u = minu; u <= maxu; u++ ) {
                    int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                    int inputboardrowoffset = inputRow * gInputBoardSize;
                    int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                    for( int v = minv; v <= maxv; v++ ) {
                        int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                        thissum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                    }
                }
            }
            _myPixelSums[pixel] += thissum;
//            if( globalId == 0 ) results[0] = _upstreamBoard[pixel];
        }
    }
    for( int pixel = 0; pixel < pixelsPerThread; pixel++ ) {
        const int virtualLocalId = localId + pixel * workgroupSize;
        if( virtualLocalId < gOutputBoardSizeSquared ) {
            float sum = _myPixelSums[pixel];
            #ifdef BIASED
                sum += biases[outPlane];
            #endif
            // results are organized like [imageid][filterid][row][col]
            int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + virtualLocalId;
            results[resultIndex ] = ACTIVATION_FUNCTION(sum);
            // results[resultIndex ] = 123;
            //if( globalId == 0 ) results[0] += 0.000001f + _perPixelSums[0];
        }
    }
}
#endif
#endif

