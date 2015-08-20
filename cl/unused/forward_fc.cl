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
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif


// each thread handles one filter, ie globalId as [n][inputplane][filterId]
// output1: [n][inputplane][filter][filterrow]
// output2: [n][inputplane][filter]
#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void reduce_rows(const int batchSize, global float const *output1, global float*output2) {
    const int globalId = get_global_id(0);
    const int n = globalId / gNumInputPlanes / gNumFilters;
    if (n >= batchSize) {
        return;
    }
    const int filterId = globalId % gNumFilters;
    float sum = 0;
    global const float *output1Col = output1 + globalId * gFilterSize;
    for (int filterRow = 0; filterRow < gFilterSize; filterRow++) {
        sum += output1Col[filterRow];
    }
    output2[globalId] = sum;
}
#endif

// each thread handles one filter, ie globalId as [n][filterId]
// output2: [n][inputplane][filter]
// output: [n][filter]
#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void reduce_inputplanes(const int batchSize, global float const *output2, global float*output) {
    const int globalId = get_global_id(0);
    const int n = globalId / gNumFilters;
    if (n >= batchSize) {
        return;
    }
    const int filterId = globalId % gNumFilters;
    float sum = 0;
    global const float *output2Col = output2 + globalId * gNumInputPlanes;
    for (int inputPlane = 0; inputPlane < gNumInputPlanes; inputPlane++) {
        sum += output2Col[inputPlane];
    }
    // activate...
    output[globalId] = ACTIVATION_FUNCTION(sum);
}
#endif

#ifdef gOutImageSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroupid [n][outputplane]
// localid: [filterrow][filtercol]
//  each thread iterates over: [inplane]
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputimagesize (mandatory)
//   outputSize == 1
//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32
//   inputimagesize around 19, not too small
#if gFilterSize == gInputSize && gPadZeros == 0
void kernel forward_filter_matches_inimage(const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *output,
    local float *_upstreamImage, local float *_filterImage) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    float sum = 0;
    for (int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gUpstreamNumPlanes + upstreamPlane) * gUpstreamImageSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numUpstreamsPerThread; i++) {
            int thisOffset = workgroupSize * i + localId;
            if (thisOffset < gUpstreamImageSizeSquared) {
                _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];
            }
        }
        const int filterGlobalOffset = (outPlane * gUpstreamNumPlanes + upstreamPlane) * gFilterSizeSquared;
        for (int i = 0; i < numFilterPixelsPerThread; i++) {
            int thisOffset = workgroupSize * i + localId;
            if (thisOffset < gFilterSizeSquared) {
                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < gOutImageSizeSquared) {
            for (int u = minu; u <= maxu; u++) {
                int inputRow = outputRow + u + (gPadZeros ? 0 : gHalfFilterSize);
                int inputimagerowoffset = inputRow * gUpstreamImageSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for (int v = minv; v <= maxv; v++) {
                    int inputCol = outputCol + v + (gPadZeros ? 0 : gHalfFilterSize);
                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // output are organized like [imageid][filterid][row][col]
    int resultIndex = (n * gNumOutPlanes + outPlane) * gOutImageSizeSquared + localId;
    if (localId < gOutImageSizeSquared) {
        output[resultIndex ] = ACTIVATION_FUNCTION(sum);
//        output[resultIndex ] = 123;
    }
}
#endif
#endif
#endif


