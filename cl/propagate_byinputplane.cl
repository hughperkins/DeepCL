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
// results: [n][filterId][outRow][outCol][inputPlane]
// need to later reduce results over: [inputPlane]
void kernel propagate_byinputplane( const int batchSize,
      global const float *images, global const float *filters, 
    global float *results,
    local float *_inputPlane, local float *_filterPlanes ) {
//    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int localId = get_local_id(0);

    const int inputPlaneId = workgroupId;
    const int numLoops = ( gNumFilters * gOutputImageSize + workgroupSize - 1 ) / workgroupSize;
    const int numFilterCopyLoops = ( gFilterSizeSquared + gOutputImageSize - 1 ) / gOutputImageSize;
    const int numImageCopyLoops = ( gInputImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
    for( int loop = 0; loop < numLoops; loop++ ) {
        const int loopLocalId = localId + loop * workgroupSize;
        const int filterId = loopLocalId / gOutputImageSize;
        const int outRow = loopLocalId % gOutputImageSize;
 
        // copy down our filter, we have gOutputImageSize threads to do this
        global float const *globalFilterPlane = filters +
            ( filterId * gNumInputPlanes + inputPlaneId ) * gFilterSizeSquared;
        local float *_localFilterPlane = _filterPlanes + filterId * gFilterSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numFilterCopyLoops; i++ ) {
            const int offset = i * gOutputImageSize + outRow;
            bool process = filterId < gNumFilters && offset < gFilterSizeSquared;
            if( process ) {
                _localFilterPlane[ offset ] = globalFilterPlane[ offset ];
            }
        }
        // loop over n ...
        for( int n = 0; n < batchSize; n++ ) {
            // copy down our imageplane, we have workgroupSize threads to do this
            barrier(CLK_LOCAL_MEM_FENCE);
            global float const *globalImagePlane = images +
                ( n * gNumInputPlanes + inputPlaneId ) * gInputImageSizeSquared;
            for( int i = 0; i< numImageCopyLoops; i++ ) {
                const int offset = i * workgroupSize + localId;
                if( offset < gInputImageSizeSquared ) {
                    _inputPlane[ offset ] = globalImagePlane[ offset ];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // calc results for each [outrow][outcol]
            bool filterPlaneOk = filterId < gNumFilters;
            for( int outCol = 0; outCol < gOutputImageSize; outCol++ ) {
                float sum = 0;
                for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
                    int inRow = outRow + filterRow;
                    #if gPadZeros == 1
                        inRow -= gHalfFilterSize;
                    #endif
                    bool rowOk = filterPlaneOk && inRow >= 0 && inRow < gInputImageSize;
                    for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                        int inCol = outCol + filterCol;
                        #if gPadZeros == 1
                            inCol -= gHalfFilterSize;
                        #endif
                        bool process = rowOk && inCol >= 0 && inCol < gInputImageSize;
                        if( process ) {
                            float imageValue = _inputPlane[ inRow * gInputImageSize + inCol ];
                            float filterValue = _localFilterPlane[ filterRow * gFilterSize + filterCol ];
                            sum += imageValue * filterValue;
                        }
                    }
                }
                if( filterId < gNumFilters ) {
                    // [n][filterId][outRow][outCol][inputPlane]
                    int resultIndex = ( ( ( n
                        * gNumFilters + filterId )
                        * gOutputImageSize + outRow )
                        * gOutputImageSize + outCol )
                        * gNumInputPlanes + inputPlaneId;
                    results[resultIndex] = sum;
                    //if( globalId == 2 ) results[0] = resultIndex;
//                    results[resultIndex] = outRow;
                }
//                results[localId] = _localFilterPlane[localId];
            }
        }
    }
}

