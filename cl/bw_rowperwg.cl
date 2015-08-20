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
// local: errorimage: outputSize
//        imageimage: inputSize
// output weight changes: [outputPlane][inputPlane][filterRow][filterCol][outRow]
void kernel backprop_weights( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *gradOutput, global const float *images, 
        global float *weightChanges,
        #ifdef BIASED
             global float *biasWeightChanges,
        #endif
        local float *_errorImage, local float *_imageImage
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int inputRow = workgroupId % gInputSize;
    const int outputPlane = (workgroupId / gInputSize) / gInputPlanes;
    const int inputPlane = (workgroupId / gInputSize) % gInputPlanes;

    // weightchanges:     [outputPlane][inputPlane][filterRow][filterCol][outRow]
    //       aggregate over:  [outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for (int n = 0; n < batchSize; n++) {
        int upstreamImageGlobalOffset = (n * gInputPlanes + inputPlane) * gInputSizeSquared;
        // need to fetch the image, but it's bigger than us, so will need to loop...
        const int numLoopsForUpstream = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numLoopsForUpstream; i++) {
            int thisOffset = i * workgroupSize + localId;
            if (thisOffset < gInputSizeSquared) {
                _imageImage[thisOffset] = images[ upstreamImageGlobalOffset + thisOffset ];
            }
        }
        int resultImageGlobalOffset = (n * gNumFilters + outputPlane) * gOutputSizeSquared;
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
                int inputRow = outRow - gMargin + filterRow;
                for (int outCol = 0; outCol < gOutputSize; outCol++) {
                    int inputCol = outCol - gMargin + filterCol;
                    bool proceed = inputRow >= 0 && inputCol >= 0 && inputRow < gInputSize
                        && inputCol < gInputSize;
                    if (proceed) {
                        int resultIndex = outRow * gOutputSize + outCol;
                        float error = _errorImage[resultIndex];
                        int upstreamDataIndex = inputRow * gInputSize + inputCol;
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
    bool writeBias = inputPlane == 0 && localId == 0;
    if (writeBias) {
        biasWeights[outputPlane] -= learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outputPlane][inputPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

