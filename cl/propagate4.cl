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
// workgroup id organized like: [n][filterid]
// local id organized like: [outrow][outcol]
// each thread iterates over: [upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// results are organized like [n][filterid][outrow][outcol]
void kernel propagate_4_by_n_outplane_smallercache( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterCube ) {
    #define globalId ( get_global_id(0) )

    #define localId ( get_local_id(0) )
    #define workgroupId ( get_group_id(0) )
//    const int workgroupSize = get_local_size(0);
    const int effectiveWorkgroupId = workgroupId / gPixelsPerThread;
    const int pixel = workgroupId % gPixelsPerThread;
    const int effectiveLocalId = localId + pixel * gWorkgroupSize;
    const int n = effectiveWorkgroupId / gNumFilters;
    const int outPlane = effectiveWorkgroupId % gNumFilters;

    const int outputRow = effectiveLocalId / gOutputBoardSize;
    const int outputCol = effectiveLocalId % gOutputBoardSize;

    float sum = 0;
    for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {
        { // these parentheses are an attempt to reduce register pressure...
            // note to self: probably should make workgroupSize a #define...
            const int numUpstreamsPerThread = ( gInputBoardSizeSquared + gWorkgroupSize - 1 ) / gWorkgroupSize;
            int thisUpstreamBoardOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numUpstreamsPerThread; i++ ) {
                int thisOffset = gWorkgroupSize * i + localId;
                if( thisOffset < gInputBoardSizeSquared ) {
                    _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
                }
            }
        }
        {
            const int numFilterPixelsPerThread = ( gFilterSizeSquared + gWorkgroupSize - 1 ) / gWorkgroupSize;
            const int filterGlobalOffset = ( outPlane * gInputPlanes + upstreamPlane ) * gFilterSizeSquared;
            for( int i = 0; i < numFilterPixelsPerThread; i++ ) {
                int thisOffset = gWorkgroupSize * i + localId;
                if( thisOffset < gFilterSizeSquared ) {
                    _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if( effectiveLocalId < gOutputBoardSizeSquared ) {
            for( int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++ ) {
                // trying to reduce register pressure...
                #if gPadZeros == 1
                    #define inputRow ( outputRow + u )
                #else
                    #define inputRow ( outputRow + u + gHalfFilterSize )
                #endif
                int inputboardrowoffset = inputRow * gInputBoardSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk = inputRow >= 0 && inputRow < gInputBoardSize;
                for( int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++ ) {
                    #if gPadZeros == 1
                        #define inputCol ( outputCol + v )
                    #else
                        #define inputCol ( outputCol + v + gHalfFilterSize )
                    #endif
                    bool process = rowOk && inputCol >= 0 && inputCol < gInputBoardSize;
                    if( process ) {
                            sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                    }
                }
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // results are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + effectiveLocalId;
    if( localId < gOutputBoardSizeSquared ) {
        results[resultIndex ] = ACTIVATION_FUNCTION(sum);
    }
    // results[resultIndex ] = 123;
    //if( globalId == 0 ) results[0] += 0.000001f + _perPixelSums[0];
}
#endif
#endif

