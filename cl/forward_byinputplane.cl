// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// concept:
// - load same input plane from each image
// - hold filter plane for this input plane, for all filters
// - reduce afterwards
// local memory for one plane from each filter of 64c7 = 64 * 7 * 7 * 4 = 12.5KB
// local memory for one single input plane = 19 * 19 * 4 = 1.4KB
// => seems ok?
// workgroupid: [inputPlaneId]
// localid: [filterId][outRow] (if this is more than workgroupsize, we should reuse some threads...)
// iterate over: [n][outCol]
// output: [n][filterId][outRow][outCol][inputPlane]
// need to later reduce output over: [inputPlane]
void kernel forward_byinputplane(const int batchSize,
      global const float *images, global const float *filters, 
    global float *output,
    local float *_inputPlane, local float *_filterPlanes) {
//    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int localId = get_local_id(0);

    const int inputPlaneId = workgroupId;
    const int numLoops = (gNumFilters * gOutputSize + workgroupSize - 1) / workgroupSize;
    const int numFilterCopyLoops = (gFilterSizeSquared + gOutputSize - 1) / gOutputSize;
    const int numImageCopyLoops = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    for (int loop = 0; loop < numLoops; loop++) {
        const int loopLocalId = localId + loop * workgroupSize;
        const int filterId = loopLocalId / gOutputSize;
        const int outRow = loopLocalId % gOutputSize;
 
        // copy down our filter, we have gOutputSize threads to do this
        global float const *globalFilterPlane = filters +
            (filterId * gNumInputPlanes + inputPlaneId) * gFilterSizeSquared;
        local float *_localFilterPlane = _filterPlanes + filterId * gFilterSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numFilterCopyLoops; i++) {
            const int offset = i * gOutputSize + outRow;
            bool process = filterId < gNumFilters && offset < gFilterSizeSquared;
            if (process) {
                _localFilterPlane[ offset ] = globalFilterPlane[ offset ];
            }
        }
        // loop over n ...
        for (int n = 0; n < batchSize; n++) {
            // copy down our imageplane, we have workgroupSize threads to do this
            barrier(CLK_LOCAL_MEM_FENCE);
            global float const *globalImagePlane = images +
                (n * gNumInputPlanes + inputPlaneId) * gInputSizeSquared;
            for (int i = 0; i< numImageCopyLoops; i++) {
                const int offset = i * workgroupSize + localId;
                if (offset < gInputSizeSquared) {
                    _inputPlane[ offset ] = globalImagePlane[ offset ];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // calc output for each [outrow][outcol]
            bool filterPlaneOk = filterId < gNumFilters;
            for (int outCol = 0; outCol < gOutputSize; outCol++) {
                float sum = 0;
                for (int filterRow = 0; filterRow < gFilterSize; filterRow++) {
                    int inRow = outRow + filterRow;
                    #if gPadZeros == 1
                        inRow -= gHalfFilterSize;
                    #endif
                    bool rowOk = filterPlaneOk && inRow >= 0 && inRow < gInputSize;
                    for (int filterCol = 0; filterCol < gFilterSize; filterCol++) {
                        int inCol = outCol + filterCol;
                        #if gPadZeros == 1
                            inCol -= gHalfFilterSize;
                        #endif
                        bool process = rowOk && inCol >= 0 && inCol < gInputSize;
                        if (process) {
                            float imageValue = _inputPlane[ inRow * gInputSize + inCol ];
                            float filterValue = _localFilterPlane[ filterRow * gFilterSize + filterCol ];
                            sum += imageValue * filterValue;
                        }
                    }
                }
                if (filterId < gNumFilters) {
                    // [n][filterId][outRow][outCol][inputPlane]
                    int resultIndex = (( (n
                        * gNumFilters + filterId)
                        * gOutputSize + outRow)
                        * gOutputSize + outCol)
                        * gNumInputPlanes + inputPlaneId;
                    output[resultIndex] = sum;
                    //if (globalId == 2) output[0] = resultIndex;
//                    output[resultIndex] = outRow;
                }
//                output[localId] = _localFilterPlane[localId];
            }
        }
    }
}

