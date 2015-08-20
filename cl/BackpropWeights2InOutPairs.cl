// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// not specifci to this kernel, but basically we have to convolve one plane of the forward output
// with one plane of each forward input, to get one filter plane change output plane
// (and do this for each example, sum them together)

// eg plane f from each output: 128 * 28 * 28 * 4 = 401KB
// plane i from each input: 128 * 28 * 28 * 4 = 401KB
// plane i from filter f: 5 * 5 * 4 = 100 bytes...
// plane i from all filters: 5 * 5 * 4 * 8 = 800 bytes (ok :-))
// all planes from all filters eg: 5 * 5 * 4 * 8 * 1 = 800 bytes (ok :-))
//
// in forward, filter plane i of filter f:
// convolves with plane i from each input cube
// to contribute to plane f of each output
// so, for backprop, need to take plane i from each input cube, and plane f
// from each output cube, and convolve together to get changes to plane i of filter f

// concept this kernel:
// we process blocks of several input and output plane pairs together
// we structure the blocks to maximize sharing between the pairs, whilst keeping the number
// of pairs down
// see prototyping/blocking.cpp for an example of calculating this cost (we could also
// including imagesize too of course)

// workgroupId: [outBlockId][inBlockId]
// localId: [filterId][inputPlane][filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
// local: errorimage: outputSize * outputSize
//        imageimage: inputSize * inputSize
void kernel backprop_floats_withscratch_dobias( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *gradOutput, global const float *images, 
        global float *weights,
        #ifdef BIASED
             global float *biasWeights,
        #endif
        local float *_errorImage, local float *_imageImage
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int outBlockId = workgroupId / gInPerBlock;
    const int inBlockId = workgroupId % gInPerBlock;

    const int localBlockId = localId / gFilterSizeSquared;
    const int localInBlock = localBlockId % gInPerBlock;
    const int localOutBlock = localBlockId / gInPerBlock;

    const int inputPlane = inBlockId * gInPerBlock + localInBlock;
    const int outputPlane = outBlockId * gOutPerBlock + localOutBlock;

    const int localLinearPos = localId % gFilterSizeSquared;
    const int filterRow = localLinearPos / gFilterSize;
    const int filterCol = localLinearPos % gFilterSize;


    for (int outPlane = 
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for (int n = 0; n < batchSize; n++) {
        int upstreamImageGlobalOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        // need to fetch the image, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numLoopsForUpstream; i++) {
            int thisOffset = i * workgroupSize + localId;
            if (thisOffset < gInputSizeSquared) {
                _imageImage[thisOffset] = images[ upstreamImageGlobalOffset + thisOffset ];
            }
        }
        int resultImageGlobalOffset = (n * gNumFilters + outPlane) * gOutputSizeSquared;
        int numLoopsForOutput = (gOutputSizeSquared + workgroupSize - 1) / workgroupSize;
        for (int i = 0; i < numLoopsForOutput; i++) {
            int thisOffset = i * workgroupSize + localId;
            if (thisOffset < gOutputSizeSquared) {
                _errorImage[thisOffset ] = gradOutput[resultImageGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < gFilterSizeSquared) {
            for (int outRow = 0; outRow < gOutputSize; outRow++) {
                int upstreamRow = outRow - gMargin + filterRow;
                for (int outCol = 0; outCol < gOutputSize; outCol++) {
                    int upstreamCol = outCol - gMargin + filterCol;
                    bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize
                        && upstreamCol < gInputSize;
                    if (proceed) {
                        int resultIndex = outRow * gOutputSize + outCol;
                        float error = _errorImage[resultIndex];
                        int upstreamDataIndex = upstreamRow * gInputSize + upstreamCol;
                        float upstreamResult = _imageImage[upstreamDataIndex];
                        thiswchange += upstreamResult * error;
    #ifdef BIASED
                        thisbiaschange += error;
    #endif
                    }
                }
            }
        }
    }
    if (localId < gFilterSizeSquared) {
        weights[ workgroupId * gFilterSizeSquared + localId ] -= learningRateMultiplier * thiswchange;
    }
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && localId == 0;
    if (writeBias) {
        biasWeights[outPlane] -= learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

