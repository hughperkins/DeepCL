// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// workgroupId: [outputPlane][inputPlane]
// localId: [filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
// local: errorimage: outputSize * outputSize
//        imageimage: inputSize * inputSize
// specific characteristic: load one stripe of each image at a time,
// so we dont run out of memory
// number of stripes set in: gNumStripes
// note that whilst we can stripe the gradOutput simply, 
// we actually need to add a half-filter widthed additional few rows
// onto the images stripe, otherwise we will be missing data
//   we will call the size of the non-overlapping image stripes: gInputStripeInnerSize
//      the outersize, including the two margins is: gInputStripeOuterSize
//      of course, the first and last stripes will be missing a bit off the top/bottom, where the 
//      corresponding outer margin would be
void kernel backprop_floats_withscratch_dobias_striped( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *gradOutput, global const float *images, 
        global float *gradWeights,
        #ifdef BIASED
             global float *gradBiasWeights,
        #endif
        local float *_errorStripe, local float *_imageStripe
 ) {
    // gHalfFilterSize
    // gInputSize
    //
    // gInputStripeMarginRows => basically equal to gHalfFilterSize
    // gInputStripeInnerNumRows = gInputSize / gNumStripes
    // gInputStripeOuterNumRows = gInputStripeInnerNumRows + 2 * gHalfFilterSize  (note: one row less than
    //                                                         if we just added gFilterSize)
    // gInputStripeInnerSize = gInputStripeInnerNumRows * gInputSize
    // gInputStripeOuterSize = gInputStripeOuterNumRows * gInputSize
    // gInputStripeMarginSize = gInputStripeMarginRows * gInputSize
    //
    // gOutputStripeNumRows
    // gOutputStripeSize

    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    const int numLoopsForImageStripe = (gInputStripeOuterSize + workgroupSize - 1) / workgroupSize;
    const int numLoopsForErrorStripe = (gOutputSizeSquared + workgroupSize - 1) / workgroupSize;
    for (int n = 0; n < batchSize; n++) {
        const int imageImageGlobalOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        const int imageImageGlobalOffsetAfter = imageImageGlobalOffset + gInputSizeSquared;
        const int errorImageGlobalOffset = (n * gNumFilters + outPlane) * gOutputSizeSquared;
        const int errorImageGlobalOffsetAfter = errorImageGlobalOffset + gOutputSizeSquared;
        for (int stripe = 0; stripe < gNumStripes; stripe++) {
            const int imageStripeInnerOffset = imageImageGlobalOffset + stripe * gInputStripeInnerSize;
            const int imageStripeOuterOffset = imageStripeInnerOffset - gInputStripeMarginSize;
            // need to fetch the image, but it's bigger than us, so will need to loop...
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = 0; i < numLoopsForImageStripe; i++) {
                int thisOffset = i * workgroupSize + localId;
                int thisGlobalImagesOffset = imageStripeOuterOffset + thisOffset;
                bool process = thisOffset < gInputStripeOuterSize 
                    && thisGlobalImagesOffset >= imageImageGlobalOffset 
                    && thisGlobalImagesOffset < imageImageGlobalOffsetAfter;
                if (process) {
                    _imageStripe[thisOffset] = images[ thisGlobalImagesOffset ];
                }
            }
            int errorStripeOffset = errorImageGlobalOffset + stripe * gOutputStripeSize;
            for (int i = 0; i < numLoopsForErrorStripe; i++) {
                int thisOffset = i * workgroupSize + localId;
                int globalErrorsOffset = errorStripeOffset + thisOffset;
                bool process = thisOffset < gOutputStripeSize 
                    && globalErrorsOffset < errorImageGlobalOffsetAfter;
                if (process) {
                    _errorStripe[thisOffset ] = gradOutput[globalErrorsOffset];
                }
            }
            const int stripeOutRowStart = stripe * gOutputStripeNumRows;
            const int stripeOutRowEndExcl = stripeOutRowStart + gOutputStripeNumRows;
            barrier(CLK_LOCAL_MEM_FENCE);
//            if (localId == 13) {
//                for (int i = 0; i < 12; i++) {
//                    gradWeights[100 + stripe * 12 + i ] = _errorStripe[i * gOutputSize];
//                }
//                for (int i = 0; i < 20; i++) {
//                    gradWeights[200 + stripe * 20 + i ] = _imageStripe[i * gInputSize];
//                }
//            }
            if (localId < gFilterSizeSquared) {
                for (int outRow = stripeOutRowStart; outRow < stripeOutRowEndExcl; outRow++) {
                    int upstreamRow = outRow - gMargin + filterRow;
                    for (int outCol = 0; outCol < gOutputSize; outCol++) {
                        int upstreamCol = outCol - gMargin + filterCol;
                        bool proceed = 
                            upstreamRow >= 0 && upstreamCol >= 0 
                            && upstreamRow < gInputSize && upstreamCol < gInputSize
                            && outRow < gOutputSize;
                        if (proceed) {
                            int resultIndex = outRow * gOutputSize + outCol;
                            float error = _errorStripe[resultIndex - stripe * gOutputStripeSize];
                            int upstreamDataIndex = upstreamRow * gInputSize + upstreamCol;
                            float upstreamResult = _imageStripe[upstreamDataIndex +  gInputStripeMarginSize
                                        - stripe * gInputStripeInnerSize ];
                            thiswchange += upstreamResult * error;
        #ifdef BIASED
                            thisbiaschange += error;
        #endif
                        }
                    }
                }
            }
        }
    }
    if (localId < gFilterSizeSquared) {
        gradWeights[ workgroupId * gFilterSizeSquared + localId ] = learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;
    if (writeBias) {
        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;
    }
#endif
    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}

