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
// concept:
//  we want to share each input example across multiple filters
//   but an entire filter plane is 19*19*4 = 1.4KB
//   so eg 500 filter planes is 500* 1.4KB = 700KB, much larger than local storage
//   of ~43KB
//  - we could take eg 16 filters at a time, store one filter plane from each in local storage,
//  and then bring down one example plane at a time, into local storage, during iteration over n
//  - here though, we are going to store one row from one plane from each filter, 
//  and process against one row, from same plane, from each example
//  so each workgroup will have one thread per filterId, eg 351 threads
//    each thread will add up over its assigned row
//  then, later we need to reduce over the rows
//   ... and also over the input planes?
//
// workgroupid [inputplane][filterrow]
// localid: [filterId]
//  each thread iterates over: [n][filtercol]
//  each thread is assigned to: one row, of one filter
//  workgroup is assigned to: same row, from each input plane
// local memory: one row from each output, = 128 * 19 * 4 = 9.8KB
//             1 * input row = "0.076KB"
// results1 structured as: [n][inputplane][filter][row], need to reduce again after
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputboardsize (mandatory)
//   inputboardsize == 19
//   filtersize == 19
//   outputBoardSize == 1
//   lots of outplanes/filters, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32-128
//   inputboardsize around 19, not too small
#if (gFilterSize == gInputBoardSize) && (gPadZeros == 0)
void kernel propagate_fc_workgroup_perrow( const int batchSize,
    global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results1,
    local float *_imageRow, local float *_filterRows ) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int localId = get_local_id(0);

    const int inputPlaneId = workgroupId / gFilterSize;
    const int filterRowId = workgroupId % gFilterSize;

    const int filterId = localId;

    // first copy down filter row, which is per-thread, so we have to copy it all ourselves...
    global const float *filterRow = filters 
        + filterId * gNumInputPlanes * gFilterSizeSquared
        + inputPlaneId * gFilterSizeSquared
        + filterRowId * gFilterSize;
//    filterRow = filters + localId * gNumInputPlanes * gFilterSizeSquared;
    local float *_threadFilterRow = _filterRows + localId * gFilterSize;
//    filterRow = filters;
    for( int i = 0; i < gFilterSize; i++ ) {
        _threadFilterRow[i] = filterRow[i];
    }
    #ifdef BIASED
    const float bias = biases[filterId];
    #endif
    const int loopsPerExample = ( gInputBoardSize + workgroupSize - 1 ) / workgroupSize;
    // now loop over examples...
    for( int n = 0; n < batchSize; n++ ) {
        // copy down example row, which is global to all threads in workgroup
        // hopefully should be enough threads....
        // but we should check anyway really, since depends on number of filters configured,
        // not on relative size of filter and input board
        global const float *exampleRow = images 
            + n * gNumInputPlanes * gInputBoardSizeSquared
            + inputPlaneId * gInputBoardSizeSquared
            + filterRowId * gInputBoardSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int loop = 0; loop < loopsPerExample; loop++ ) {
            int offset = loop * workgroupSize + localId;
            if( offset < gInputBoardSize ) {
                _imageRow[offset] = exampleRow[offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // add up the values in our row...
        float sum = 0;
        for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
            sum += _imageRow[ filterCol ] * _threadFilterRow[ filterCol ];
        }
        #ifdef BIASED
        sum += bias;
        #endif
        // note: dont activate yet, since need to reduce again
        // results structured as: [n][filter][inputplane][filterrow], need to reduce again after
        if( localId < gNumFilters ) {
            results1[ n * gNumInputPlanes * gNumFilters * gFilterSize
                + inputPlaneId * gFilterSize
                + filterId * gNumInputPlanes * gFilterSize + filterRowId ] = sum;
        }
    }
}
#endif
#endif

kernel void reduce_segments( const int numSegments, const int segmentLength, 
        global float const *in, global float* out ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
//    const int workgroupId = get_group_id(0);
//    const int localSegment = localId / segmentLength;
    const int segmentId = globalId;

    if( segmentId >= numSegments ) {
        return;
    }

    float sum = 0;
    global const float *segment = in + segmentId * segmentLength;
    for( int i = 0; i < segmentLength; i++ ) {
        sum += segment[i];
    }
    out[segmentId] = sum;
}

#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void activate( const int N, global float *inout ) {
    const int globalId = get_global_id(0);
    if( globalId >= N ) {
        return;
    }
    inout[globalId] = ACTIVATION_FUNCTION( inout[globalId] );
}
#endif

// each thread handles one filter, ie globalId as [n][inputplane][filterId]
// results1: [n][inputplane][filter][filterrow]
// results2: [n][inputplane][filter]
#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void reduce_rows( const int batchSize, global float const *results1, global float*results2 ) {
    const int globalId = get_global_id(0);
    const int n = globalId / gNumInputPlanes / gNumFilters;
    if( n >= batchSize ) {
        return;
    }
    const int filterId = globalId % gNumFilters;
    float sum = 0;
    global const float *results1Col = results1 + globalId * gFilterSize;
    for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
        sum += results1Col[filterRow];
    }
    results2[globalId] = sum;
}
#endif

// each thread handles one filter, ie globalId as [n][filterId]
// results2: [n][inputplane][filter]
// results: [n][filter]
#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void reduce_inputplanes( const int batchSize, global float const *results2, global float*results ) {
    const int globalId = get_global_id(0);
    const int n = globalId / gNumFilters;
    if( n >= batchSize ) {
        return;
    }
    const int filterId = globalId % gNumFilters;
    float sum = 0;
    global const float *results2Col = results2 + globalId * gNumInputPlanes;
    for( int inputPlane = 0; inputPlane < gNumInputPlanes; inputPlane++ ) {
        sum += results2Col[inputPlane];
    }
    // activate...
    results[globalId] = ACTIVATION_FUNCTION(sum);
}
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


