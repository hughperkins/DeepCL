// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

// globalId: [outPlane][inputPlane][filterRow][filterCol]
// per-thread iteration: [n][outputRow][outputCol]
void kernel backprop_floats(const float learningRateMultiplier,
        const int batchSize, 
         global const float *gradOutput, global const float *images, 
        global float *gradWeights
        #ifdef BIASED
            , global float *gradBiasWeights
        #endif
 ) {
    int globalId = get_global_id(0);
    if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize) {
        return;
    }

    int IntraFilterOffset = globalId % gFilterSizeSquared;
    int filterRow = IntraFilterOffset / gFilterSize;
    int filterCol = IntraFilterOffset % gFilterSize;

    int filter2Id = globalId / gFilterSizeSquared;
    int outPlane = filter2Id / gInputPlanes;
    int upstreamPlane = filter2Id % gInputPlanes;

    float thiswchange = 0;
    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for (int n = 0; n < batchSize; n++) {
        for (int outRow = 0; outRow < gOutputSize; outRow++) {
            int upstreamRow = outRow - gMargin + filterRow;
            for (int outCol = 0; outCol < gOutputSize; outCol++) {
                int upstreamCol = outCol - gMargin + filterCol;
                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize
                    && upstreamCol < gInputSize;
                if (proceed) {
                    int resultIndex = (( n * gNumFilters 
                              + outPlane) * gOutputSize
                              + outRow) * gOutputSize
                              + outCol;
                    float error = gradOutput[resultIndex];
                    int upstreamDataIndex = (( n * gInputPlanes 
                                     + upstreamPlane) * gInputSize
                                     + upstreamRow) * gInputSize
                                     + upstreamCol;
                    float upstreamResult = images[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * error;
                    thiswchange += thisimagethiswchange;
    #ifdef BIASED
                    thisbiaschange += error;
    #endif
                }
            }
        }
    }
    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;
    if (writeBias) {
        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;
    }
#endif
}



