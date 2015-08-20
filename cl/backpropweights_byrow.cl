// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// reminder:
// - for backprop weights, we take one plane from one image, convolve with one plane from the output
//   and reduce over n

// concept:
// - here, we process only single row from the input/output cube (same row from each)
//   and then we will need to reduce the resulting weight changes over the rows, in a separate kernel
// - this assumes that the filter cubes are small, so reducing over 32 or so of them is not a big task

// this isnt expected to give good performance, but it paves the way for creating workgroups with
// multiple pairs of input/output planes in, which might reduce memory copying from global
// filters themselves are fairly small, and plasuibly easy to reduce?

// here, we will use one workgroup for one row of a single pair of input/output planes
// and sum over n
// workgroup: [outputPlane][inputPlane][outputRow]
// localid: [filterRow][filterCol]
// weightChanges1: [outputPlane][inputPlane][filterRow][filterCol][outputRow]
// gradBiasWeights1: [outputPlane][outputRow]
kernel void backprop_weights(const float learningRateMultiplier, const int batchSize,
    global float const *gradOutput, global float const *input, global float *restrict gradWeights1,
    #ifdef BIASED
         global float *restrict gradBiasWeights1,
    #endif
    local float *restrict _errorRow, local float *restrict _inputRow) {
    #define globalId (get_global_id(0))
    #define workgroupId (get_group_id(0))
    #define localId (get_local_id(0))
    
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;
    const int outputRow = workgroupId % gOutputSize;
    #define outInCombo (workgroupId / gOutputSize)
    const int outputPlane = outInCombo / gNumInputPlanes;
    const int inputPlane = outInCombo % gNumInputPlanes;

    const int thisInputRow = outputRow - gMargin; // + filterRow;

    float thiswchange = 0.0f;
    #ifdef BIASED
        float thisbiaschange = 0.0f;
    #endif
    for (int n = 0; n < batchSize; n++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        // copy down the gradOutput row...
        {
            global float const*gradOutputRow = gradOutput + 
                (( n
                    * gNumOutputPlanes + outputPlane)
                    * gOutputSize + outputRow)
                    * gOutputSize;
            if (localId < gOutputSize) { // assume we have enough threads for now... should fix later
                _errorRow[ localId ] = gradOutputRow[ localId ];
            }
        }
        // copy down the input row
        {
            global float const*inputRowData = input +
                (( n
                    * gNumInputPlanes + inputPlane)
                    * gInputSize + thisInputRow)
                    * gInputSize;
            if (localId < gInputSize) { // assume we have enough threads for now... should fix later
                _inputRow[ localId ] = inputRowData[ localId ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int outputCol = 0; outputCol < gOutputSize; outputCol++) {
            const int inputCol = outputCol - gMargin + filterCol;
            if (inputRow >= 0 && inputRow < gInputSize && inputCol >= 0 && inputCol < gInputSize) {
                if (localId < gFilterSizeSquared) {
                    thiswchange += _inputRow[ inputCol ] * _errorRow[ outputCol ];
                    #ifdef BIASED
                        thisbiaschange += _errorRow[ outputCol ];
                    #endif
                }
            }
        }
    }

    if (workgroupId == 0 && localId == 0) {
        gradWeights1[0] = _inputRow[0];
        gradWeights1[1] = _inputRow[1];
    }

    if (localId < gFilterSizeSquared) {
        #define weightsIndex (( (outInCombo \
            * gFilterSizeSquared) + localId \
            * gOutputSize) + outputRow)
        //gradWeights1[ weightsIndex ] -= learningRateMultiplier * thiswchange;
        //gradWeights1[weightsIndex] = 123.0f;
    }
    #ifdef BIASED
        if (inputPlane == 0 && localId == 0) {
            gradBiasWeights1[outputPlane * gOutputSize + outputRow ] = learningRateMultiplier * thisbiaschange;
        }
    #endif
}

